#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import keras
import pickle
from cab_cnn import get_model
import math


MAX_POOLING_SIZE = 5
SR = int(22050/MAX_POOLING_SIZE)
SNIPPET_LENGTH = 30
input_size = SNIPPET_LENGTH * SR
NUM_CLASSIFIERS = 40

model = keras.models.load_model('model00000030.h5')

with open('progressive.pkl', 'rb') as f:
    songs = pickle.load(f)

predictions = []
for song_no, song in enumerate(songs):
    song = np.array([max(song[i:i + 5]) for i in range(0, len(song), 5)]).reshape((-1, 1))
    length = len(song) / SR
    num_snippets = math.ceil(length / SNIPPET_LENGTH)
    displacement = (len(song) - (SNIPPET_LENGTH * SR)) // (num_snippets - 1)
    song_snippets = []
    for i in range(num_snippets):
        song_snippets.append(song[displacement * i:displacement * i + SNIPPET_LENGTH * SR])
    song_snippets = np.array(song_snippets)
    prediction = np.sum(model.predict(song_snippets)) / num_snippets
    predictions.append(prediction)
    if song_no % 5 == 0:
        print(f'Song {song_no} of {len(songs)}')
print(keras.metrics.binary_accuracy(np.zeros(len(predictions)), predictions))
