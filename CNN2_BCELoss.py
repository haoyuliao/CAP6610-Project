#!/usr/bin/env python
# coding: utf-8

# In[32]:


#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
import torch, torchvision, os, cv2
from torch.utils.data import random_split, Dataset
from torch.nn import Sigmoid, Tanh, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Sigmoid, BatchNorm2d, Flatten, ConvTranspose2d
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

images = np.load('LogImages.npy') #Rade the trianing data.
images = np.moveaxis(images, -1, 1) #Reshape channeL from [B, H, W, C] to [B, C, H, W]
labels = np.load('Labels.npy') #Rade the trianing data. 
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
trainImg, testImg = random_split(images, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
trainLab, testLab = random_split(labels, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test

print(trainImg[0].shape)
print(trainLab[25])


trainData = [] 
for i in range(len(trainImg)):
    trainData.append([torch.tensor(trainImg[i], dtype=torch.float32), torch.tensor(trainLab[i],dtype=torch.float32)])
trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=5)

testData = []
for i in range(len(testImg)):
    testData.append([torch.tensor(testImg[i], dtype=torch.float32), torch.tensor(testLab[i],dtype=torch.float32)])
testLoader = torch.utils.data.DataLoader(testData, shuffle=False, batch_size=5)


# In[33]:


net = Sequential(
            # Defining 1st 2D convolution layer
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1), #200@3 
            BatchNorm2d(16),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining 2nd 2D convolution layer
            Conv2d(16, 8, kernel_size=3, stride=1, padding=1), #100@3 
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining 3rd 2D convolution layer
            Conv2d(8, 4, kernel_size=3, stride=1, padding=1), #50@3 
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining 4th 2D convolution layer
            Conv2d(4, 2, kernel_size=3, stride=1, padding=1), #25@3 
            BatchNorm2d(2),
            ReLU(inplace=True),
            Flatten(),
            Linear(2 * 25 * 25, 1),
            Sigmoid()
        )

net = net.cuda()
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(net)


# In[34]:


#%%time
n_epoch = 10000
for epoch in range(n_epoch):  # loop over the dataset multiple times
#epoch = 0
#while True:
    train_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].cuda(), data[1].cuda() #Reg
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        labels = torch.reshape(labels, (-1,))
        outputs = torch.reshape(outputs, (-1,))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
    logStr = 'Train epoch: %d, Loss: %.10f' % (epoch + 1, train_loss/ (i+1))
    print(logStr)
    epoch += 1

print('Finished Training')


# In[46]:


# PATH = './CNN2_SGDlr0.001_BCEL_EP10000_72%.pth'
# torch.save(net.state_dict(), PATH)


# In[44]:


correct = 0
total = 0
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for data in trainLoader:
        images, labels = data[0].cuda(), data[1].cuda()
        outputs = net(images)
        predicted = torch.round(outputs)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        

print('Accuracy of the network on the %s test images: %d %%' % (total,
    100 * correct / total))

print(confusion_matrix)


# In[48]:


correct = 0
total = 0
nb_classes = 2
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for data in testLoader:
        images, labels = data[0].cuda(), data[1].cuda()
        outputs = net(images)
        predicted = torch.round(outputs)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        

print('Accuracy of the network on the %s test images: %d %%' % (total,
    100 * correct / total))

print(confusion_matrix)


# In[ ]:




