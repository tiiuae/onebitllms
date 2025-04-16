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
import copy
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, integrations

def _is_param_to_not_quantize(param_name):
    param_to_not_quantize = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "input_layernorm",
        "post_attention_layernorm",
    ]

    for p in param_to_not_quantize:
        if p in param_name:
            return True
    return False

def _weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-05)
    u = (w * scale).round().clamp_(-1, 1)
    return u, scale

def quantize_to_1bit(input_checkpoint_path: str, output_checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(input_checkpoint_path)

    model = AutoModelForCausalLM.from_pretrained(
        input_checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    state_dict = model.state_dict()
    state_dict_quant = copy.deepcopy(state_dict)

    for param_name, param_value in state_dict.items():
        if _is_param_to_not_quantize(param_name):
            continue


        param_value, param_scale = _weight_quant(param_value)
        param_value = integrations.pack_weights(param_value)

        state_dict_quant[param_name] = param_value
        state_dict_quant[param_name.replace("weight", "weight_scale")] = (
            param_scale.view(1)
        )

    modules_to_not_convert = integrations.get_keys_to_not_convert(model)

    model = integrations.replace_with_bitnet_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
    )

    model.load_state_dict(state_dict_quant, assign=True, strict=True)
    model.save_pretrained(output_checkpoint_path, safe_serialization=True)

    config = AutoConfig.from_pretrained(input_checkpoint_path)

    config.is_bitnet_config = True

    config.quantization_config = {
        "modules_to_not_convert": modules_to_not_convert,
        "quant_method": "bitnet",
    }
    config.pretraining_tp = 1
    config.save_pretrained(output_checkpoint_path)
    tokenizer.save_pretrained(output_checkpoint_path)

    return output_checkpoint_path


def convert_to_bf16(input_checkpoint_path: str, output_checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(input_checkpoint_path)

    model = AutoModelForCausalLM.from_pretrained(
        input_checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    state_dict = model.state_dict()
    state_dict_quant = copy.deepcopy(state_dict)

    for param_name, param_value in state_dict.items():
        if _is_param_to_not_quantize(param_name):
            continue

        param_value, param_scale = _weight_quant(param_value)
        param_value = param_value / param_scale

        state_dict_quant[param_name] = param_value

    model.load_state_dict(state_dict_quant, assign=True, strict=True)
    model.save_pretrained(output_checkpoint_path, safe_serialization=True)

    tokenizer.save_pretrained(output_checkpoint_path)
    return output_checkpoint_path
