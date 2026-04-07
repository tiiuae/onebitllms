# Copyright 2026 Falcon-LLM Team. All rights reserved.
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
import sys
from argparse import ArgumentParser

from ..utils import quantize_to_1bit, convert_to_bf16

CLI_CMD_MAPPING = {
    'convert': quantize_to_1bit,
    'convert_in_bf16': convert_to_bf16,
}

def main():
    if len(sys.argv) == 1:
        raise ValueError(f"You need to specify a command from: {CLI_CMD_MAPPING.keys()}, e.g.: `onebitllms convert` ")
    command_name = sys.argv[1]

    if command_name not in CLI_CMD_MAPPING:
        raise ValueError(f"Invalid command name. Supported command names are: {CLI_CMD_MAPPING.keys()}")
    
    command_function = CLI_CMD_MAPPING[command_name]
    print(f"Executing: {command_function}")
    command_function(*sys.argv[2:])
    # print(command_name)






