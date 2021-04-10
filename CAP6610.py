#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import random_split

def getData():
    
    #load extracted data from files
    folder = "ExtractDataset"
    images = np.load(folder+'/LogImages.npy') #Rade the trianing data.
    images = np.moveaxis(images, -1, 1) #Reshape channeL from [B, H, W, C] to [B, C, H, W]
    labels = np.load(folder+'/Labels.npy') #Rade the trianing data. 
    labels = labels.reshape(labels.shape[0],1)
    # images = images.astype(np.float32)
    # labels = labels.astype(np.int)
    
    print(labels.shape)
    labels2D = np.zeros((labels.shape[0],2))
    for i in range(len(labels)):
        lab = labels[i]
        if lab == 0:
            labels2D[i,0] = 1
        if lab == 1:
            labels2D[i,1] = 1
    lengths = [round(len(images)*0.8), round(len(images)*0.2)]
    print(lengths)
    
    #perform training/testing data splits
    trainImg, testImg = random_split(images, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
    trainLab, testLab = random_split(labels, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
    
    print(trainImg[0].shape)
    print(trainLab[25])
    
    #get training data
    trainData = [] 
    for i in range(len(trainImg)):
        trainData.append([torch.tensor(trainImg[i], dtype=torch.float32), torch.tensor(trainLab[i],dtype=torch.float32)])
    #trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=5)
    
    #get test data
    testData = []
    for i in range(len(testImg)):
        testData.append([torch.tensor(testImg[i], dtype=torch.float32), torch.tensor(testLab[i],dtype=torch.float32)])
    #testLoader = torch.utils.data.DataLoader(testData, shuffle=False, batch_size=5)

    #return trainLoader, testLoader
    return trainData, testData