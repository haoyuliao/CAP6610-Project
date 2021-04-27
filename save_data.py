#!/usr/bin/env python3
import librosa
import librosa.display
from os import walk
from matplotlib import pyplot as plt
import pickle

other_songs_paths = []
top_of_the_pops_paths = []
progressive_paths = []

for (directory, _, filenames) in walk('Dataset/Not_Progressive_Rock/Other_Songs/'):
    for filename in filenames:
        other_songs_paths.append('/'.join([directory, filename]))

for (directory, _, filenames) in walk('Dataset/Not_Progressive_Rock/Top_Of_The_Pops/'):
    for filename in filenames:
        top_of_the_pops_paths.append('/'.join([directory, filename]))

for (directory, _, filenames) in walk('Dataset/Progressive_Rock_Songs'):
    for filename in filenames:
        progressive_paths.append('/'.join([directory, filename]))

SR = 22050
other_songs_data = []
top_of_the_pops_data = []
progressive_data = []

for path in other_songs_paths:
    x, sr = librosa.load(path, sr=SR)
    other_songs_data.append(x)
    print(f'Loaded {path} with duration {len(x)/sr}')

for path in top_of_the_pops_paths:
    x, sr = librosa.load(path, sr=SR)
    top_of_the_pops_data.append(x)
    print(f'Loaded {path} with duration {len(x)/sr}')

for path in progressive_paths:
    x, sr = librosa.load(path, sr=SR)
    progressive_data.append(x)
    print(f'Loaded {path} with duration {len(x) / sr}')

with open('other_songs.pkl', 'wb') as f:
    pickle.dump(other_songs_data, f)

with open('top_of_the_pops.pkl', 'wb') as f:
    pickle.dump(top_of_the_pops_data, f)

with open('progressive.pkl', 'wb') as f:
    pickle.dump(progressive_data, f)
