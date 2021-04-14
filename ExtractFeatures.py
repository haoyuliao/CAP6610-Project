#!/usr/bin/env python3
# coding: utf-8

#documentation page for librosa module
#https://librosa.org/doc/main/auto_examples/plot_display.html

import librosa, audioread
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, cv2, warnings
from pydub import AudioSegment

#functioncall for creating output directory if missing
def createDirIfMissing(dir):
        if not os.path.isdir(dir):
                os.makedirs(dir)
                        
#these folders must be created before running script
folders=['Progressive_Rock_Songs','Not_Progressive_Rock/Other_Songs','Not_Progressive_Rock/Top_Of_The_Pops']

saveFolders = ['ExtractDataset/prog/', 'ExtractDataset/nonprog/']

#check for existence of output save folders
for folder in saveFolders:
        createDirIfMissing(folder)

#1 for prog; 0 for nonprog
database = {'idx':[],'ReName':[],'Style':[],'Label':[],'StartDur (Sec)':[],'EndDur (Sec)':[],'Duration (Sec)':[]
            ,'OriName':[], 'DbMax':[],'DbMin':[]}
saveArr = []
saveImgArr = []
saveImgArrLog=[]
labels = []


chroma_stft = []
chroma_cqt = []
chroma_cens = []
melspectogram = []
mfcc = []
rms = []
spectral_centroid = []
spectral_bandwidth = []
spectral_contrast = []
spectral_flatness = []
spectral_rolloff = []
poly_features = []
tonnetz = []
zero_crossing_rate = []
tempogram = []
fourier_tempogram = []


filesName = []
for folder in folders:
    if folder == 'Progressive_Rock_Songs':
        saveFolder = saveFolders[0]
        reName = 'Prog-'
        style = 'Prog'
        idx = 0
        label = 1

    if folder == 'Not_Progressive_Rock/Other_Songs':
        saveFolder = saveFolders[1]
        reName = 'NonProgOther-'
        style = 'NonProg'
        idx = 0
        label = 0
        
    if folder == 'Not_Progressive_Rock/Top_Of_The_Pops':
        reName = 'NonProgTopPops-'
        label = 0
    
    filesNames = os.listdir('./Dataset/'+folder)
    for i in range(len(filesNames)):
        fileName = filesNames[i]
        print(i)
        print(fileName)
        audio_path = './Dataset/'+folder+'/'+fileName
        formalName = fileName.split('.')[-1]
        FilExt = fileName.split('.')[-1]
        if FilExt != 'mp3':
            flac_audio = AudioSegment.from_file('./Dataset/'+folder+'/'+fileName, FilExt)
            fileName = fileName.replace(fileName.split('.')[-1],'')+'mp3'
            audio_path = './FlatToMP3/'+folder+'/'+fileName
            createDirIfMissing('./FlatToMP3/'+folder)
            flac_audio.export(audio_path, format="mp3")    
        print(fileName)
        saveName = reName+str(idx)
        totalDur = audioread.audio_open(audio_path).duration

        plt.cla()
        plt.clf()
        plt.close('all')
        plt.figure(1, figsize=(14, 5))
        
        #load file in librosa
        x , sr = librosa.load(audio_path, sr=22050, offset=totalDur/2-30, duration=60.0) 
        
        #display Spectrogram
        X = librosa.stft(x, hop_length=256, n_fft=4096)
        Xdb = librosa.amplitude_to_db(abs(X), ref=np.max)
        
        #get features
        feature1 = librosa.feature.chroma_stft(x,sr)        
        feature2 = librosa.feature.chroma_cqt(x,sr)
        feature3 = librosa.feature.chroma_cens(x,sr)
        feature4 = librosa.feature.melspectrogram(x,sr)
        feature5 = librosa.feature.mfcc(x,sr)
        feature6 = librosa.feature.rms(x,sr)
        feature7 = librosa.feature.spectral_centroid(x,sr)
        feature8 = librosa.feature.spectral_bandwidth(x,sr)
        feature9 = librosa.feature.spectral_contrast(x,sr)
        feature10 = librosa.feature.spectral_flatness(x)
        feature11 = librosa.feature.spectral_rolloff(x,sr)
        feature12 = librosa.feature.poly_features(x,sr)
        feature13 = librosa.feature.tonnetz(x,sr)
        feature14 = librosa.feature.zero_crossing_rate(x)
        feature15 = librosa.feature.tempogram(x,sr)
        feature16 = librosa.feature.fourier_tempogram(x,sr)
        
        
        print(Xdb.shape)
        if Xdb.shape[1] != 5168:
            continue
        

        #librosa.display.specshow(Xdb, hop_length=256, sr=sr, x_axis='time', y_axis='hz', cmap='gray_r') #, vmin=-50, vmax=50, 
        librosa.display.specshow(Xdb, hop_length=256, sr=sr, x_axis='time', y_axis='hz') #, vmin=-50, vmax=50, 
        cb=plt.colorbar(format="%+2.f dB")
        
        plt.savefig(saveFolder+saveName+'-withAix.png')

        plt.axis('off')
        cb.remove()
        plt.draw() #update plot
        plt.savefig(saveFolder+saveName+'.png')
        #plt.show()       
    
        ###Resize each img size into 200x200 for CNN training
        readImg1 = cv2.imread(saveFolder+saveName+'.png')
        resizeImg1 = cv2.resize(readImg1,(200,200))
        cv2.imwrite(saveFolder+saveName+'-Resize.png', resizeImg1)
        #########################################################
        
        ###Log img###############################################
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.figure(2, figsize=(14, 5))
        librosa.display.specshow(Xdb, hop_length=256, sr=sr, x_axis='time', y_axis='log') #, vmin=-50, vmax=50,
        cb=plt.colorbar(format="%+2.f dB")
        
        plt.savefig(saveFolder+saveName+'-LogwithAix.png')

        plt.axis('off')
        cb.remove()
        plt.draw() #update plot
        plt.savefig(saveFolder+saveName+'-Log.png')
        #plt.show()       
    
        ###Resize each img size into 200x200 for CNN training
        readImg2 = cv2.imread(saveFolder+saveName+'-Log.png')
        resizeImg2 = cv2.resize(readImg2,(200,200))
        cv2.imwrite(saveFolder+saveName+'-LogResize.png', resizeImg2)
        ##############################################################
        ##############################################################
        
        saveArr.append(Xdb)
        labels.append(label)
        filesName.append(fileName)
        saveImgArr.append(resizeImg1)
        saveImgArrLog.append(resizeImg2)
        
        #add to feature arrays
        chroma_stft.append(feature1)
        chroma_cqt.append(feature2)
        chroma_cens.append(feature3)
        melspectogram.append(feature4)
        mfcc.append(feature5)
        rms.append(feature6)
        spectral_centroid.append(feature7)
        spectral_bandwidth.append(feature8)
        spectral_contrast.append(feature9)
        spectral_flatness.append(feature10)
        spectral_rolloff.append(feature11)
        poly_features.append(feature12)
        tonnetz.append(feature13)
        zero_crossing_rate.append(feature14)
        tempogram.append(feature15)
        fourier_tempogram.append(feature16)
        
        database['idx'].append(idx)
        database['ReName'].append(saveName)
        database['OriName'].append(fileName)
        database['Style'].append(style)
        database['Label'].append(label)
        database['StartDur (Sec)'].append(totalDur/2-30)
        database['EndDur (Sec)'].append(totalDur/2-30+60)
        database['Duration (Sec)'].append(totalDur)
        database['DbMax'].append(Xdb.max())
        database['DbMin'].append(Xdb.min())
        idx+=1

    #break
np.save('./ExtractDataset/DBHzarray.npy', saveArr)
np.save('./ExtractDataset/Imgages.npy', saveImgArr)
np.save('./ExtractDataset/LogImages.npy', saveImgArrLog) 
np.save('./ExtractDataset/Labels.npy', labels)
np.save('./ExtractDataset/FilesName.npy', filesName) 

createDirIfMissing('./ExtractDataset/Features/')
np.save('./ExtractDataset/Features/chroma_stft.npy',chroma_stft)
np.save('./ExtractDataset/Features/chroma_cqt.npy',chroma_cqt)
np.save('./ExtractDataset/Features/chroma_cens.npy',chroma_cens)
np.save('./ExtractDataset/Features/melspectogram.npy',melspectogram)
np.save('./ExtractDataset/Features/mfcc.npy',mfcc)
np.save('./ExtractDataset/Features/rms.npy',rms)
np.save('./ExtractDataset/Features/spectral_centroid.npy',spectral_centroid)
np.save('./ExtractDataset/Features/spectral_bandwidth.npy',spectral_bandwidth)
np.save('./ExtractDataset/Features/spectral_contrast.npy',spectral_contrast)
np.save('./ExtractDataset/Features/spectral_flatness.npy',spectral_flatness)
np.save('./ExtractDataset/Features/spectral_rolloff.npy',spectral_rolloff)
np.save('./ExtractDataset/Features/poly_features.npy',poly_features)
np.save('./ExtractDataset/Features/tonnetz.npy',tonnetz)
np.save('./ExtractDataset/Features/zero_crossing_rate.npy',zero_crossing_rate)
np.save('./ExtractDataset/Features/tempogram.npy',tempogram)
np.save('./ExtractDataset/Features/fourier_tempogram.npy',fourier_tempogram)
        
DatabaseToPd = pd.DataFrame(data=database)
DatabaseToPd.to_excel('Database.xlsx', index=True)

