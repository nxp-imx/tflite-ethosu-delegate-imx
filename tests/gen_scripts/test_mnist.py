#
# Copyright 2020-2022 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

def generate_model(path):
    # Download MNIST dataset.
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    images = tf.cast(train_images[0], tf.float32)/255.0
    mnist_ds = tf.data.Dataset.from_tensors((images)).batch(1)

    def representative_dataset():
        for input_value in mnist_ds.take(100):
            yield [input_value]

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Define the model architecture
    model = keras.Sequential([
         keras.layers.Flatten(input_shape=(28, 28)),
    # Optional: You can replace the dense layer above with the convolution layers below to get higher accuracy.
         keras.layers.Reshape(target_shape=(28, 28, 1)),
         keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
         keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
         keras.layers.MaxPooling2D(pool_size=(2, 2)),
         keras.layers.Dropout(0.25),
         keras.layers.Flatten(input_shape=(28, 28)),
         keras.layers.Dense(128, activation=tf.nn.relu),
         keras.layers.Dropout(0.5),

         keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # Train the digit classification model
    model.fit(train_images, train_labels, epochs=1)

    # Convert Keras model to TF Lite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # quantize the model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

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
