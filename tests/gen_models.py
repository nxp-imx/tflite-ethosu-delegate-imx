#!/usr/bin/python3
#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import fnmatch
import sys
from importlib import import_module
from multiprocessing import Process

model_path = "tflite_models"
if not os.path.exists(model_path):
    os.mkdir(model_path)

cases = []
package = 'gen_scripts'
for f in os.listdir(package):
    if fnmatch.fnmatch(f, "test_*.py"):
        m = import_module(f".{f[:-3]}", package=package)
        m.file = f
        cases.append(m)

print("Generating tflite models ... ")
for m in cases:
    print(m)
    print("processing ", m.file)
    #m.generate_model(model_path)
    #keras.backend.clear_session()
    p = Process(target=m.generate_model, args=(model_path,))
    p.start()
    p.join()
