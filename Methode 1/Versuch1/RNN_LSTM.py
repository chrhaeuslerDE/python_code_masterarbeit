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
from keras.layers import Dense, Dropout, Activation, GaussianNoise
from keras.layers import LSTM
from keras import callbacks, optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

history = callbacks.History()

adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=11, min_lr=0.00001)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=80, 
                                         mode='auto')

# load the dataset
print('Loading data...')
dataframeTrain = pandas.DataFrame(pandas.read_csv('../train.txt', sep = ' '))
dataframeTest = pandas.DataFrame(pandas.read_csv('../test.txt', sep = ' '))
datasetTr = dataframeTrain.values
datasetTr = datasetTr.astype('float32')
datasetTe = dataframeTest.values
datasetTe = datasetTe.astype('float32')


# split into input (X) and output (Y) variables
trainX = datasetTr[0:45916,6:26]
trainY = datasetTr[0:45916,27]
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

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#Respahe Y to (samples,1)
trainY = numpy.reshape(trainY, (len(trainY),1))


# create and fit the LSTM network
def small_model():
    model = Sequential()

    model.add(LSTM(
        20,
        input_shape=(1,20),
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GaussianNoise(0.2))

    model.add(LSTM(
        10,
        return_sequences=False))
    model.add(Dropout(0.1))
    model.add(GaussianNoise(0.1))

    model.add(Dense(1))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer=adam)
    return model



model = small_model()

model.fit(trainX, trainY, epochs=500, batch_size=125,validation_split=0.1 ,\
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
