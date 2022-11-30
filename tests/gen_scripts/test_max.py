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
    SP = (2, 2)
    def representative_dataset():
      for _ in range(100):
        x = tf.random.uniform(SP) * 255 
        y = tf.random.uniform(SP) * 255 
        yield [x, y]

    x = keras.layers.Input(shape=SP)
    y = keras.layers.Input(shape=SP)
    out = keras.layers.Maximum()([x, y])

    model = keras.models.Model(inputs=[x, y], outputs=out)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # quantize the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # convert model
    tflite_quant_model = converter.convert()

    module_name = str(os.path.basename(__file__)).split('.')[0]
    model_path = os.path.join(path, f'{module_name}_uint8.tflite')
    # save model to a file
    with open(model_path, 'wb') as f:
        f.write(tflite_quant_model)

if __name__ == '__main__':
    path = "model"
    os.makedirs(path, exist_ok=True)
    generate_model(path)