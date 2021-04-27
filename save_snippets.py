#!/usr/bin/env python3
import math
import pickle
from matplotlib import pyplot as plt
import librosa
import librosa.display
import numpy as np


with open('other_songs.pkl', 'rb') as f:
    other_songs = pickle.load(f)

with open('top_of_the_pops.pkl', 'rb') as f:
    top_of_the_pops = pickle.load(f)

with open('progressive.pkl', 'rb') as f:
    progressive = pickle.load(f)

SNIPPET_LENGTH = 30
MAX_SNIPPETS = 15
MAX_POOLING_SIZE = 5
SR = int(22050/MAX_POOLING_SIZE)
non_progressive_data = []
progressive_data = []

for song in [*top_of_the_pops, *other_songs]:
    song = [max(song[i:i + MAX_POOLING_SIZE]) for i in range(0, len(song), MAX_POOLING_SIZE)]
    length = len(song)/SR
    num_snippets = min(MAX_SNIPPETS, math.ceil(length / SNIPPET_LENGTH))
    displacement = (len(song) - (SNIPPET_LENGTH * SR))//(num_snippets - 1)
    for i in range(num_snippets):
        non_progressive_data.append(song[displacement * i:displacement * i + SNIPPET_LENGTH * SR])

for song in progressive:
    song = [max(song[i:i + MAX_POOLING_SIZE]) for i in range(0, len(song), MAX_POOLING_SIZE)]
    length = len(song)/SR
    num_snippets = min(MAX_SNIPPETS, math.ceil(length / SNIPPET_LENGTH))
    displacement = (len(song) - (SNIPPET_LENGTH * SR))//(num_snippets - 1)
    for i in range(num_snippets):
        progressive_data.append(song[displacement * i:displacement * i + SNIPPET_LENGTH * SR])

non_progressive_data = np.array(non_progressive_data)
progressive_data = np.array(progressive_data)

print(non_progressive_data.shape)
print(progressive_data.shape)

np.save('non_progressive_data.npy', non_progressive_data)
np.save('progressive_data.npy', progressive_data)
