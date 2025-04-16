# Copyright 2025 The Falcon-LLM Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import triton
import triton.language as tl
import torch

@triton.jit
def jitted_max(a, b):
    return tl.maximum(a, b)

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 64}),
        # triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=['hidden_dim']
)
@triton.jit
def partial_max_kernel(
    x_ptr,               # [batch_size, seq_len, hidden_dim]
    partial_max_ptr,     # [batch_size, seq_len, n_blocks]
    batch_size, seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each (batch_idx, seq_idx, block_idx), compute partial max(|x|)
    of the slice [block_idx*BLOCK_SIZE : (block_idx+1)*BLOCK_SIZE] along hidden_dim,
    then store it in partial_max[batch_idx, seq_idx, block_idx].
    """
    batch_idx = tl.program_id(0)
    seq_idx   = tl.program_id(1)
    blk_idx   = tl.program_id(2)

    n_blocks = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Out-of-range checks (optional safety)
    if batch_idx >= batch_size:
        return
    if seq_idx >= seq_len:
        return
    if blk_idx >= n_blocks:
        return

    # Slice of the hidden dimension for this block
    hidden_off = blk_idx * BLOCK_SIZE
    offsets    = tl.arange(0, BLOCK_SIZE) + hidden_off
    mask       = offsets < hidden_dim

    # Base offset for (batch_idx, seq_idx) in x
    row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim

    # Load data, compute partial max of absolute values
    x_vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
    # partial_absmax = tl.reduce(tl.abs(x_vals), axis=0, combine_fn=jitted_max)
    partial_absmax = tl.abs(x_vals)
    partial_absmax = tl.max(partial_absmax, axis=0)
    partial_absmax = tl.maximum(partial_absmax, 1e-5)

    # Store the partial max => partial_max array
    # partial_max shape is (batch_size, seq_len, n_blocks), flattened in memory
    pm_offset = (batch_idx * seq_len + seq_idx) * n_blocks + blk_idx
    tl.store(partial_max_ptr + pm_offset, partial_absmax)
    
@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 64}),
        # triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ],
    key=['hidden_dim']
)
@triton.jit
def finalize_and_quant_kernel(
    x_ptr,               # [batch_size, seq_len, hidden_dim]
    partial_max_ptr,     # [batch_size, seq_len, n_blocks]
    y_ptr,               # [batch_size, seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each (batch_idx, seq_idx):
      1) Merge partial_max => find final row-wise max
      2) scale = 127 / row_absmax
      3) Reload x (in a loop), quantize, store y
    """
    batch_idx = tl.program_id(0)
    seq_idx   = tl.program_id(1)

    if batch_idx >= batch_size:
        return
    if seq_idx >= seq_len:
        return

    # Number of blocks that covered hidden_dim
    n_blocks = (hidden_dim + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Merge partial maxima
    pm_base = (batch_idx * seq_len + seq_idx) * n_blocks
    row_absmax = 0.0

    i = 0
    while i < n_blocks:
        val_i = tl.load(partial_max_ptr + pm_base + i)
        row_absmax = tl.maximum(row_absmax, val_i)
        i += 1

    # Compute scale
    row_absmax = tl.maximum(row_absmax, 1e-5)
    scale = 127.0 / row_absmax

    # Now quantize the entire row [0..hidden_dim) in a loop
    row_offset = (batch_idx * seq_len + seq_idx) * hidden_dim
    # We'll step in increments of BLOCK_SIZE
    i = 0
    while i < n_blocks:
        hidden_off = i * BLOCK_SIZE
        offsets    = tl.arange(0, BLOCK_SIZE) + hidden_off
        mask       = offsets < hidden_dim

        x_vals = tl.load(x_ptr + row_offset + offsets, mask=mask, other=0.0)
        scaled_x  = x_vals * scale
        rounded_x = tl.floor(scaled_x + 0.5)
        quant_x   = tl.clamp(rounded_x, -128, 127) / scale

        tl.store(y_ptr + row_offset + offsets, quant_x, mask=mask)
        i += 1

@torch.compiler.disable
def activation_quant_triton(x: torch.Tensor) -> torch.Tensor:
    """
    2-step approach for row-wise max of |x| across hidden_dim.
    Then quantize to int8 range [-128..127], then dequant back to float.
    """
    batch_size, seq_len, hidden_dim = x.shape
    y = torch.empty_like(x)

    # We choose the largest block size from our autotuning config
    max_block_size = 256
    n_blocks = (hidden_dim + max_block_size - 1) // max_block_size
    # n_blocks = 1

    # Temporary buffer to store partial maxima
    partial_max = torch.empty(
        (batch_size * seq_len * n_blocks,),
        dtype=x.dtype,
        device=x.device
    )

    # 1) Partial maxima
    grid1 = (batch_size, seq_len, n_blocks)
    partial_max_kernel[grid1](x, partial_max, batch_size, seq_len, hidden_dim)

    # 2) Final merge + quant
    grid2 = (batch_size, seq_len)
    finalize_and_quant_kernel[grid2](x, partial_max, y, batch_size, seq_len, hidden_dim)

    return y