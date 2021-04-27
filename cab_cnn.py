import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MAX_POOLING_SIZE = 5
SR = int(22050/MAX_POOLING_SIZE)
SNIPPET_LENGTH = 30
input_size = SNIPPET_LENGTH * SR
NUM_CLASSIFIERS = 40


def get_model():
    inputs = keras.Input(shape=(input_size, 1))

    x = keras.layers.Conv1D(filters=16, kernel_size=4, padding='same', activation=keras.activations.relu)(inputs)
    x = keras.layers.MaxPool1D(pool_size=4, strides=2, padding='same')(x)
    x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.MaxPool1D(pool_size=4, strides=2, padding='same')(x)
    x = keras.layers.Dropout(0.15)(x)
    x = keras.layers.Conv1D(filters=32, kernel_size=10, padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.MaxPool1D(pool_size=10, strides=5, padding='same')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Conv1D(filters=128, kernel_size=10, padding='same', activation=keras.activations.relu)(x)
    x = keras.layers.MaxPool1D(pool_size=10, strides=5, padding='same')(x)

    classifiers = []
    for i in range(NUM_CLASSIFIERS):
        c_inputs = keras.Input(shape=(1323, 128))
        y = keras.layers.Dense(8, activation=keras.activations.relu)(c_inputs)
        y = keras.layers.Dense(4, activation=keras.activations.relu)(y)
        y = keras.layers.Dense(1, activation=keras.activations.sigmoid)(y)
        classifier = keras.Model(inputs=c_inputs, outputs=y, name=f'Classifier_{i+1}')
        classifiers.append(classifier)

    classifier_outputs = []
    for i in range(NUM_CLASSIFIERS):
        classifier_outputs.append(classifiers[i](x))
    classifier_outputs = keras.layers.Concatenate()(classifier_outputs)

    acb_inputs = keras.Input(shape=(1323, 128))
    y = keras.layers.Dense(160, activation=keras.activations.relu)(acb_inputs)
    y = keras.layers.Dense(80, activation=keras.activations.relu)(y)
    y = keras.layers.Dense(40, activation=keras.activations.softmax)(y)
    attention_block = keras.Model(inputs=acb_inputs, outputs=y, name=f'Attention_Block')

    acb_outputs = attention_block(x)

    outputs = keras.layers.Multiply()([classifier_outputs, acb_outputs])
    outputs = tf.reduce_sum(outputs, axis=-1)
    outputs = tf.reduce_mean(outputs, axis=-1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Adam', loss=keras.losses.binary_crossentropy, metrics=[keras.metrics.binary_accuracy])

    return model


