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
import torch
import torch.nn as nn
import torch.nn.functional as F

from onebitllms import activation_quant_triton, weight_quant_triton

class BitNetLinear(nn.Module):
    """
    Implementation of the BitNet Linear layer. The BitNet linear layer consists
    of having one linear layer with clamped weights during training and additional
    non learnable layer normalization layers

    Attributes:
        in_features (`int`):
            Number of input features of the linear layer
        out_features (`int`):
            Number of output features of the lienar layer
    """
    def __init__(self, in_features, out_features, bias: bool = False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)


    def forward(self, x: torch.Tensor):
        w = self.weight

        with torch.cuda.device(w.device):
            x_quant = x + (activation_quant_triton(x) - x).detach()
            w_quant = w + (weight_quant_triton(w) - w).detach()
        
        y = F.linear(x_quant, w_quant, bias=self.bias)
        return y
    
    def __repr__(self):
        return 'BitnetLinear(in_features={0}, out_features={1})'.format(self.in_features, self.out_features)