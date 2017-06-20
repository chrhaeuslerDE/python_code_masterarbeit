#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:56:23 2017

@author: entwicklung
"""

# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras import callbacks, optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

history = callbacks.History()

adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.0001)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=30, 
                                         mode='auto')

# load the dataset
print('Loading data...')
dataframeTrain = pandas.DataFrame(pandas.read_csv('train.txt', sep = ' '))
dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))
datasetTr = dataframeTrain.values
datasetTr = datasetTr.astype('float32')
datasetTe = dataframeTest.values
datasetTe = datasetTe.astype('float32')


# split into input (X) and output (Y) variables
trainX = datasetTr[0:45000,6:26]
trainY = datasetTr[0:45000,27]
testX = datasetTe[210:325,6:26] #Sequence 3
testY = datasetTe[210:325,27]

#Trainingsdatensatz Standardisieren
sc = StandardScaler()
sc.fit(trainX)
trainX = sc.transform(trainX)

#Testdatensatz Standardisieren
sc = StandardScaler()
sc.fit(testX)
testX = sc.transform(testX)

"""
# normalize the dataset
scalerTrainX = MinMaxScaler(feature_range=(0, 1))
trainX = scalerTrainX.fit_transform(trainX)

scalerTrainY = MinMaxScaler(feature_range=(0, 1))
trainY = scalerTrainY.fit_transform(trainY)

scalerTestX = MinMaxScaler(feature_range=(0, 1))
testX = scalerTestX.fit_transform(testX)

scalerTestY = MinMaxScaler(feature_range=(0, 1))
testY = scalerTestY.fit_transform(testY)
"""

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#Respahe Y to (samples,1)
trainY = numpy.reshape(trainY, (len(trainY),1))


# create and fit the LSTM network
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        layers[0],
        input_shape=(1,20),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
    layers[1],
    input_shape=(1,20),
    return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            layers[3]))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer="adam")
    return model


model = build_model([1, 50, 100, 1])

model.fit(trainX, trainY, epochs=250, batch_size=150,validation_split=0.05 ,\
          verbose=1, callbacks=[history, reduce_lr, early_stopping])

# make predictions
testPredict = model.predict(testX)

# plot baseline and predictions
plt.plot(testPredict)
testY = datasetTe[210:325,27]
plt.plot(testY)
plt.show()

"""
#PLot the history of the training
plt.plot(history)
plt.title("History of Epochs")
plt.show()
"""
