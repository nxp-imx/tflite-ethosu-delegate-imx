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
import tensorflow as tf
from tensorflow import keras

def generate_model(path):
    SP = [1, 4]
    def representative_dataset():
        for _ in range(100):
            data1 = tf.random.normal(SP)
            yield [data1]

    x = tf.constant(1., shape=SP[1:])
    func = tf.function(lambda x: tf.tanh(x))
    concrete_func = func.get_concrete_function(x)

    # quantize the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8

    # convert model
    tflite_quant_model = converter.convert()

    module_name = str(os.path.basename(__file__)).split('.')[0]
    model_path = os.path.join(path, f'{module_name}_int8.tflite')
    # save model to a file
    with open(model_path, 'wb') as f:
        f.write(tflite_quant_model)

if __name__ == '__main__':
    path = "model"
    os.makedirs(path, exist_ok=True)
    generate_model(path)
