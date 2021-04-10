#!/usr/bin/env python3
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import seaborn as sns

def scaleData(data):
    sc = StandardScaler()
    return sc.fit_transform(data)

def convert1dto2d(data):
    ohe = OneHotEncoder()
    return ohe.fit_transform(data.reshape(-1,1)).toarray()

def convert2dto1d(data):
    return np.argmax(data, axis = 1)

def plotCM(conMatrix,title=None):
    figure = plt.figure(figsize = (8,8))
    sns.heatmap(conMatrix, annot=True,cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()
    plt.close(figure)
    
class Classifier:
    def __init__(self):
        pass
    
class NeuralNetwork(Classifier):
    msSum = 0.0
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(120000, input_dim=120000, activation='tanh'))
        self.model.add(Dense(2, activation='softmax'))        
        self.model.compile(loss='categorical_crossentropy')
    def fit(self,X,y):
        return self.model.fit(X, convert1dto2d(y), epochs=200, batch_size=5)
    def predict(self, X):
        return self.model.predict(X)
    def getCM(self, X, y):
        y_pred = self.model.predict_classes(X)
        return tf.math.confusion_matrix(labels = y, predictions = y_pred).numpy()
    def plot(self, X, y):
        cm = self.getCM(X,y)
        plotCM(cm,title='Neural Network Split(' + str(currentSplit) + ')')
    def tabulate(self,X,y):
        y_pred = self.predict(X)
        NeuralNetwork.msSum += accuracy_score(y,convert2dto1d(y_pred))
    #def fit(self,X,y,Xt,yt):
    #    return self.model.fit(X, y, epochs=1000, batch_size=5, validation_data=(Xt,yt))
    
class FisherDiscrim(Classifier):
    msSum = 0.0
    def __init__(self):
        self.clf = LinearDiscriminantAnalysis()
    def fit(self,X,y):
        self.clf.fit(X,y)
    def plot(self,X,y):
        plot_confusion_matrix(self.clf, X, y).ax_.set_title('Fisher Discriminant Split(' + str(currentSplit) + ')')
    def tabulate(self,X,y):
        y_pred = self.clf.predict(X)
        FisherDiscrim.msSum += accuracy_score(y,y_pred)
    
class RandomForests(Classifier):
    msSum = 0.0
    def __init__(self):
        self.clf = RandomForestClassifier()
    def fit(self,X,y):
        return self.clf.fit(X,y)
    def plot(self,X,y):
        plot_confusion_matrix(self.clf, X, y).ax_.set_title('Random Forests Split(' + str(currentSplit) + ')')
    def tabulate(self,X,y):
        y_pred = self.clf.predict(X)
        RandomForests.msSum += accuracy_score(y,y_pred)

class LinearSVM(Classifier):
    msSum = 0.0
    def __init__(self):
        self.clf = svm.SVC()
    def fit(self,X,y):
        return self.clf.fit(X,y)
    def plot(self,X,y):
        plot_confusion_matrix(self.clf, X, y).ax_.set_title('Linear SVM Split(' + str(currentSplit) + ')')
    #def test(self,X,y)
    def tabulate(self,X,y):
        y_pred = self.clf.predict(X)
        LinearSVM.msSum += accuracy_score(y,y_pred)
    
class RegressionTree(Classifier):
    msSum = 0.0
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()
    def fit(self,X,y):
        return self.clf.fit(X,y)
    def plot(self,X,y):
        plot_confusion_matrix(self.clf, X, y).ax_.set_title('Regression Tree Split(' + str(currentSplit) + ')')
    def tabulate(self,X,y):
        y_pred = self.clf.predict(X)
        RegressionTree.msSum += accuracy_score(y,y_pred)
        
def getData():
    

    #load extracted data from files
    folder = "ExtractDataset"
    images = np.load(folder+'/LogImages.npy') #Rade the trianing data.
    images = images.reshape(254,120000)
    #print(images.shape)
    #images = np.moveaxis(images, -1, 1) #Reshape channeL from [B, H, W, C] to [B, C, H, W]
    #print(images.shape)
    labels = np.load(folder+'/Labels.npy') #Rade the trianing data. 
    #print(labels.shape)
    #labels = labels.reshape(labels.shape[0],1)
    #print(labels.shape)
    
    #print(labels.shape)
    '''
    labels2D = np.zeros((labels.shape[0],2))
    for i in range(len(labels)):
        lab = labels[i]
        if lab == 0:
            labels2D[i,0] = 1
        if lab == 1:
            labels2D[i,1] = 1
    lengths = [round(len(images)*0.8), round(len(images)*0.2)]
    #print(lengths)
    '''
    ##perform training/testing data splits
    #trainImg, testImg = random_split(images, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
    #trainLab, testLab = random_split(labels, lengths ,generator=torch.random.manual_seed(42)) #Shuffle data with random seed 42 before split train and test
    
    #print(images.shape)
    #print(labels.shape)
    
    #print(trainImg[0].shape)
    #print(trainLab[25])
    
    return images, labels

def main():
    
    #load dataset
    images, labels = getData()
    
    #scale data by removing mean and scaling to unit variance
    X = images
    #X = scaleData(images)
        
    #get classification labels
    y = labels
    
    #create list of classifiers to use
    Classifiers = [FisherDiscrim, RandomForests, LinearSVM, RegressionTree]
    
    #perform on 10 different splits
    for i in range(10):
        
        #perform split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        #set global var for plot's split identification in plot title
        global currentSplit
        currentSplit = i + 1
        
        #run split on each classifier
        for cl in Classifiers:
            
            #instantate
            classifier = cl()
            
            #fit
            classifier.fit(X_train,y_train)
            
            #plot confusion matrix for test data
            classifier.plot(X_test, y_test)
            
            #tabulate misclassification errors using accuracy
            classifier.tabulate(X_test,y_test)
            
    print()
    print("Average Accuracy of Different Classifiers")
    for cl in Classifiers:
        avg = cl.msSum / 10
        print(cl.__name__ + ": " + str(avg))


if __name__ == "__main__":
    #run main script
    main()