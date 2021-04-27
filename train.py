#!/usr/bin/env python3
from cab_cnn import get_model
import numpy as np
import tensorflow as tf
import keras


model = get_model()

non_progressive = np.load('non_progressive_data.npy')
progressive = np.load('progressive_data.npy')

train_non_progressive = non_progressive[:int(non_progressive.shape[0] * 0.85)]
val_non_progressive = non_progressive[int(non_progressive.shape[0] * 0.85):]
train_progressive = progressive[:int(progressive.shape[0] * 0.85)]
val_progressive = progressive[int(progressive.shape[0] * 0.85):]

train_data = np.concatenate([train_non_progressive, train_progressive], axis=0)
train_targets = np.concatenate([np.zeros(train_non_progressive.shape[0]), np.ones(train_progressive.shape[0])], axis=0)
val_data = np.concatenate([val_non_progressive, val_progressive], axis=0)
val_targets = np.concatenate([np.zeros(val_non_progressive.shape[0]), np.ones(val_progressive.shape[0])], axis=0)

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_targets))
val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_targets))
train_dataset = train_dataset.shuffle(10000).batch(32)
val_dataset = val_dataset.shuffle(10000).batch(32)

checkpoint = keras.callbacks.ModelCheckpoint('model{epoch:08d}.h5', period=5)
model.fit(train_dataset, epochs=100, callbacks=[checkpoint], validation_data=val_dataset)