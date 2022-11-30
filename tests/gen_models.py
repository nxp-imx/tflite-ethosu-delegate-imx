#!/usr/bin/python3
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
