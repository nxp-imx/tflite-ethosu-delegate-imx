#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import os
import argparse
import tflite_runtime.interpreter as tflite
from ethosu.vela import vela

def verify(cpu_model, vela_model, delegate_path):
    cpu_interpreter = tflite.Interpreter(cpu_model)
    cpu_interpreter.allocate_tensors()

    ext_delegate = [tflite.load_delegate(delegate_path)]
    vela_interpreter = tflite.Interpreter(vela_model, experimental_delegates=ext_delegate)
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
    required=True,
    help='Delegate library path')
args = parser.parse_args()

for model in os.listdir(args.input):
    cpu_model = os.path.join(args.input, model)
    vela_model = vela.convert(cpu_model)
    print("Model %33s" % model)
    ret = verify(cpu_model, vela_model, args.delegate)
    print("Offline compile, model %33s %s" % (model, "PASS" if ret else "FAIL"))
    ret = verify(cpu_model, cpu_model, args.delegate)
    print("Online compile, model %33s %s" % (model, "PASS" if ret else "FAIL"))
