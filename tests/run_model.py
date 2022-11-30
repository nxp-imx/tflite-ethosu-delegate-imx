#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import os
import argparse
import tflite_runtime.interpreter as tflite
from ethosu.vela import vela

def verify(cpu_model, vela_model, delegate_path):
    cpu_interpreter = tflite.Interpreter(cpu_model)
    cpu_interpreter.allocate_tensors()

    if (delegate_path):
        ext_delegate = [tflite.load_delegate(delegate_path)]
        vela_interpreter = tflite.Interpreter(vela_model, experimental_delegates=ext_delegate)
    else:
        vela_interpreter = tflite.Interpreter(vela_model)
    vela_interpreter.allocate_tensors()

    cpu_inputs = cpu_interpreter.get_input_details()
    vela_inputs = vela_interpreter.get_input_details()
    # prepare input tensors
    for i in range(len(cpu_inputs)):
        dtype = cpu_inputs[i]['dtype']
        shape = cpu_inputs[i]['shape']
        x = np.random.randint(low=np.iinfo(dtype).min,
                   high=np.iinfo(dtype).max, size=shape, dtype=dtype)
        cpu_interpreter.set_tensor(cpu_inputs[i]['index'], x)
        vela_interpreter.set_tensor(vela_inputs[i]['index'], x)

    # invoke the inference
    cpu_interpreter.invoke()
    vela_interpreter.invoke()

    cpu_outputs = cpu_interpreter.get_output_details()
    vela_outputs = vela_interpreter.get_output_details()
    for i in range(len(cpu_outputs)):
        cpu_data = cpu_interpreter.get_tensor(cpu_outputs[i]['index'])
        vela_data = vela_interpreter.get_tensor(vela_outputs[i]['index'])
        if (cpu_data != vela_data).any():
            return False

    return True

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='tflite_models',
    help='Input tflite models path')
parser.add_argument(
    '-d',
    '--delegate',
    default='',
    help='Delegate library path')
args = parser.parse_args()

for model in os.listdir(args.input):
    cpu_model = os.path.join(args.input, model)
    vela_model = vela.convert(cpu_model)
    ret = verify(cpu_model, vela_model, args.delegate)
    print("Model %33s %s" % (model, "PASS" if ret else "FAIL"))
