# CAP6610-Project

## Install Dependencies ##
* Python3
* librosa
* matplotlib
* pandas
* opencv-python
* pydub
* torch
* torchvision
* openpyxl
* tensorflow
* keras

## Install Tools ##
* ffmpeg

  https://www.ffmpeg.org/

## Setup ##
1. Clone repository
```
git clone https://github.com/ryonakennedy/CAP6610-Project
```
2. Change to working directory from clone
```
cd CAP6610-Project
```
3. Install project requirements (linux example)
```
pip3 install -r requirements.txt
```
```
apt-get install ffmpeg
```
4. Create folder for music files used to extract features from
```
mkdir Dataset
```
5. Copy music files into Dataset folder retaining music folder hierarchy
(make sure only audio files are in directories or script may halt/fail)
```
Dataset
├── Not_Progressive_Rock
└── Progressive_Rock_Songs
Dataset/Not_Progressive_Rock/
├── Other_Songs
└── Top_Of_The_Pops
```
## Method 1 ##
### Extracting Data for Classifiers ###
For extracting features from sound files
```
./ExtractFeatures.py
```
### Running Classifiers ###
* CNN2_BCELoss
```
./CNN2_BCELoss.py
```
* Fisher Discriminant, Random Forests, Linear SVM, Regression Tree
```
./SciKitLearnClassifiers.py
```
## Method 2 ##
### Extracting Data for Classifiers ###
Extract Raw Data
```
./save_data.py
```
Extract snippets from Raw Data
```
./save_snippets.py
```
### Running Classifiers ###
* Classifier-Attention-Based CNN
Train
Evaluate
