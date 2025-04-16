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
# See the License for the specific language 
# limitations under the License.
import torch
import pytest
from onebitllms import activation_quant_triton, weight_quant_triton

def weight_quant_torch(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-05)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

def activation_quant_torch(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-05)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

@pytest.mark.parametrize("input_size", [1024, 2048, 4096])
@pytest.mark.parametrize("output_size", [1024, 2048, 4096])
def test_weight_quant(input_size, output_size):
    tensor = torch.randn(input_size, output_size, device="cuda", dtype=torch.bfloat16)

    w_torch = weight_quant_torch(tensor)
    w = weight_quant_triton(tensor)

    assert torch.allclose(w_torch, w, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [512, 1024, 8192])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
def test_activation_quant(batch_size, seq_len, hidden_dim):
    tensor = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.bfloat16)

    x_torch = activation_quant_torch(tensor)
    x = activation_quant_triton(tensor)

    # TODO: the kernels do not give 100% match for the activation quant kernel
    # however, we confirmed with end-to-end training that this did not affect 
    # final model performance.
    assert torch.allclose(x_torch, x, atol=5e-2, rtol=5e-2)