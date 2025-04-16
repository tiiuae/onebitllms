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
def abs_mean_kernel(
    x_ptr,         # float32 or float16 pointer to [n_elems]
    partial_ptr,   # float32 pointer to [grid_size] for partial sums
    n_elems,       # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    """
    Each block:
      1) Loads up to BLOCK_SIZE elements (depending on mask).
      2) Takes absolute value and does a local reduction (sum).
      3) Writes the sum to partial_ptr[block_id].
    After the kernel, we do partial_ptr.sum() / n_elems in Python to get the mean.
    """

    # This block's ID and offsets
    block_id = tl.program_id(0)
    offsets  = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < n_elems

    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Take absolute value
    x = tl.abs(x)

    # Step 1: in-block reduction, dimension = BLOCK_SIZE
    # a typical warp-level approach is:
    #   sum_x = tl.sum(x, axis=0)
    # but for older Triton or more advanced usage, you might do a manual tree-reduce.
    # The simplest built-in is:
    sum_x = tl.sum(x, axis=0)

    # Write out the partial sum to partial_ptr
    # each block writes exactly one element
    tl.store(partial_ptr + block_id, sum_x)


@triton.jit
def weight_quant_kernel(
    x_ptr,
    y_ptr,
    scale,
    n_elems,       # total number of elements
    BLOCK_SIZE: tl.constexpr
):
    """
    Quantization kernel:
      y = clamp(round(x * scale), -1, 1) / scale
    """
    block_id = tl.program_id(0)
    offsets  = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < n_elems

    # Load
    w = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Scale
    w_scaled  = w * scale
    # Round
    w_rounded = tl.floor(w_scaled + 0.5)
    # Clamp to [-1, 1]
    w_clamped = tl.clamp(w_rounded, -1, 1)
    # Dequant
    w_q = w_clamped / scale

    tl.store(y_ptr + offsets, w_q, mask=mask)

@torch.compiler.disable
def weight_quant_triton(w_2d: torch.Tensor, block_size=256):
    """
    1) Flatten w_2d to 1D
    2) Launch abs_mean_kernel => partial sums => final mean
    3) Launch weight_quant_kernel => store in output
    4) Reshape output to same shape as w_2d
    """
    assert w_2d.is_cuda, "tensor must be on CUDA"

    # Flatten for simpler indexing
    w_1d = w_2d.contiguous().view(-1)
    n_elems = w_1d.numel()

    # ---- Pass 1: compute mean(|w|)
    grid = ( (n_elems + block_size - 1) // block_size, )
    partial_sums = torch.empty(grid[0], dtype=torch.float32, device=w_2d.device)

    abs_mean_kernel[grid](
        w_1d,            # x_ptr
        partial_sums,    # partial_ptr
        n_elems,         # n_elems
        BLOCK_SIZE=block_size
    )

    # finalize the mean in python
    total_sum = partial_sums.sum()
    mean_abs = total_sum.item() / float(n_elems)
    mean_abs = max(mean_abs, 1e-5)
    scale = 1.0 / mean_abs

    # ---- Pass 2: quant
    w_out = torch.empty_like(w_1d)
    weight_quant_kernel[grid](
        w_1d,
        w_out,
        scale,
        n_elems,
        BLOCK_SIZE=block_size
    )

    # reshape back to 2D
    w_out_2d = w_out.view_as(w_2d)
    return w_out_2d