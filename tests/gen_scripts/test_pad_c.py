#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import tensorflow as tf
from tensorflow import keras

def generate_model(path):
    SP = [2, 2, 2]
    def representative_dataset():
        for i in range(255):
          data1 = tf.ones(SP)
          data1 *= (i - 128)
          yield [data1]

    paddings = tf.constant([0, 0, 0, 0, 0, 2], shape=[3, 2])

    x = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], shape=SP)
    func = tf.function(lambda x: tf.pad(x, paddings, "CONSTANT", constant_values=0.0))
    concrete_func = func.get_concrete_function(x)

    # quantize the model
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

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
