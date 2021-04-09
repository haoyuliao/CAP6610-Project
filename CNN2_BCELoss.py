#!/usr/bin/env python3
# coding: utf-8

#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#training-on-gpu
import torch, torchvision, os, cv2
from torch.utils.data import random_split, Dataset
from torch.nn import Sigmoid, Tanh, Linear, ReLU, Sequential, Conv2d, MaxPool2d, Sigmoid, BatchNorm2d, Flatten, ConvTranspose2d
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#---CNN2_BCELosss Class with Default Training Parameters---#

class CNN2_BCELoss:
    
    def __init__(self):
        
        #define the neural network
        self.net = Sequential(
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
        
        #move neural network to the gpu
        self.net = self.net.cuda()
        
        #define default criterion
        self.criterion = nn.BCELoss()
        
        #define default optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        
        #define default epochs to use for training
        #self.epochs = 10000
        
    def train(self, datas, epochs=10000):
        for epoch in range(epochs):  # loop over the dataset multiple times
        #epoch = 0
        #while True:
            train_loss = 0.0
            for i, data in enumerate(datas, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].cuda(), data[1].cuda() #Reg
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.net(inputs)
                labels = torch.reshape(labels, (-1,))
                outputs = torch.reshape(outputs, (-1,))
        
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        
                # print statistics
                train_loss += loss.item()
            logStr = 'Train epoch: %d, Loss: %.10f' % (epoch + 1, train_loss/ (i+1))
            print(logStr)
            epoch += 1
        
        print('Finished Training')
        
    def printConfusionMatrix(self, datas):
            correct = 0
            total = 0
            nb_classes = 2
            confusion_matrix = torch.zeros(nb_classes, nb_classes)
            with torch.no_grad():
                for data in datas:
                    images, labels = data[0].cuda(), data[1].cuda()
                    outputs = self.net(images)
                    predicted = torch.round(outputs)
            
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                    
            
            print('Accuracy of the network on the %s test images: %d %%' % (total,
                100 * correct / total))
            
            print(confusion_matrix)
            
#---END---CNN2_BCELosss Class with Training Parameters---#


#---Function for Loading Training / Test Data From File---#

def getData():
    

    #load extracted data from files
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
    
    #perform training/testing data splits
    trainImg, testImg = random_split(images, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
    trainLab, testLab = random_split(labels, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
    
    print(trainImg[0].shape)
    print(trainLab[25])
    
    #get training data
    trainData = [] 
    for i in range(len(trainImg)):
        trainData.append([torch.tensor(trainImg[i], dtype=torch.float32), torch.tensor(trainLab[i],dtype=torch.float32)])
    trainLoader = torch.utils.data.DataLoader(trainData, shuffle=True, batch_size=5)
    
    #get test data
    testData = []
    for i in range(len(testImg)):
        testData.append([torch.tensor(testImg[i], dtype=torch.float32), torch.tensor(testLab[i],dtype=torch.float32)])
    testLoader = torch.utils.data.DataLoader(testData, shuffle=False, batch_size=5)

    return trainLoader, testLoader

#---END---Function for Loading Training / Test Data From File---#


#function for running this file as a script
def main():
    
    #get data from file loaded
    trainLoader, testLoader = getData()
    
    #instantate CNN2_BCELoss Neural Network with default parameters
    NN = CNN2_BCELoss()
    
    #train the network with loaded training data
    NN.train(trainLoader, epochs=100)
    
    #evaluate and print confusion matrices for data
    NN.printConfusionMatrix(trainLoader)
    NN.printConfusionMatrix(testLoader)
    

#test if user is running this file as a script
if __name__ == "__main__":
    #run function defined as main
    main()


#---Refactored Code for Classless Solution---#

'''
#define the neural network
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

#move neural network to the gpu
net = net.cuda()

#define criterion
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

#define optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(net)


def train(net, criterion, optimizer, n_epoch, training):
    for epoch in range(n_epoch):  # loop over the dataset multiple times
    #epoch = 0
    #while True:
        train_loss = 0.0
        for i, data in enumerate(training, 0):
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


#function for evaluating and printing confusion matrix from data    
def printConfusionMatrix(net, datas):
    correct = 0
    total = 0
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data in datas:
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
    
def main():
    #get data from file loaded
    trainLoader, testLoader = getData()
    
    #train the neural net specifying criterion, optimizer, epochs, and training data
    train(net, criterion, optimizer, 10000, trainLoader)
    
    #print confusion matrix for training data
    printConfusionMatrix(net, trainLoader)
    
    #print confusion matrix for testing data
    printConfusionMatrix(net, testLoader)
'''
#---END---Refactored Code for Classless Solution---#


#some old unfactored code in notes below

'''
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

'''

# PATH = './CNN2_SGDlr0.001_BCEL_EP10000_72%.pth'
# torch.save(net.state_dict(), PATH)


'''
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
'''


'''
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

'''


