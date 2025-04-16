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
from typing import Optional

import torch
import torch.nn as nn

from onebitllms.layers import BitNetLinear

def replace_linear_with_bitnet_linear(model, previous_dtype: Optional[torch.dtype] = None):
    """
    """
    # Recursively replace linear layers
    if previous_dtype is None:
        previous_dtype = torch.get_default_dtype()

        model_dtype = model.dtype
        torch.set_default_dtype(model_dtype)

        previous_dtype = model_dtype

    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_bitnet_linear(module, previous_dtype=previous_dtype)
        
        # Replace nn.Linear layers, but skip 'lm_head'
        if name != 'lm_head' and isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            with torch.device(module.weight.device):
                # Create a new instance of the custom linear layer
                new_layer = BitNetLinear(in_features, out_features, bias=bias)
                # Copy weights and biases
                with torch.no_grad():
                    new_layer.weight.copy_(module.weight)
                    if bias:
                        new_layer.bias.copy_(module.bias)
            
            # Replace the layer in the model
            setattr(model, name, new_layer)
    return model