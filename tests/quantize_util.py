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
import tensorflow as tf

def gen_sample_dataset(zero_point, scale, shape):
    low = (0.0 - zero_point) * scale
    high = (1.0 - zero_point) * scale

    def representative_dataset():  
        for _ in range(256):
            v1 = np.random.uniform(low, high, (shape)).astype(np.float32)
            v2 = np.random.uniform(low, high, (shape)).astype(np.float32)
            yield [v1, v2]

    return representative_dataset        

def quantize_converter(converter, zero_point, scale, shape):
    # quantize the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = gen_sample_dataset(zero_point, scale, shape)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    return converter
