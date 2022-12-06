#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import tensorflow as tf
from tensorflow import keras

def generate_model(path):
    SP = (1, 8, 8, 3)
    def representative_dataset():
      for _ in range(100):
        data1 = tf.random.normal(list(SP))
        yield [data1]

    x = keras.layers.Input(shape=SP[1:], batch_size=1)
    out = keras.layers.Conv2D(2, 3,input_shape=SP[1:])(x)
    model = keras.models.Model(inputs=[x], outputs=out)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # quantize the model
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
