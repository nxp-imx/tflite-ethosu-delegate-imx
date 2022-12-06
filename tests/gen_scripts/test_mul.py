#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import tensorflow as tf
from tensorflow import keras
import quantize_util as qu

def generate_model(path):
    SP = [2, 2]

    x = keras.layers.Input(shape=SP)
    y = keras.layers.Input(shape=SP)

    out = keras.layers.multiply([x, y])
    model = keras.models.Model(inputs=[x, y], outputs=out)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # quantize model with zero_point, scale, shape
    converter = qu.quantize_converter(converter, 0.0, 0.025, SP)

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
