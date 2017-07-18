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

print('Loading data...')

dataframeTrain = pandas.DataFrame(pandas.read_csv('../PHM008_HERI_training.txt', sep = ' '))

#Fügt den Spalten den Feature-Namen hinzu
dataframeTrain.columns  = ['index1','index2','unit', 'cyles', 'OS1', \
'OS2', 'OS3', 's1', 's2'\
, 's3', 's4', 's5', 's6', 's7', 's8'\
, 's9', 's10', 's11', 's12', 's13', 's14'\
, 's15', 's16', 's17', 's18', 's19', 's20'\
, 's21', 'HERI1', 'HERI2', 'HERI3', 'HERI4', 'HERI5', 'HERI6','RUL']



#Speichert nur die 9 wichtigen Sensoren
trainX = dataframeTrain.drop(['index1','index2', 'unit', 'cyles', 'OS1', \
'OS2', 'OS3','s1', 's5', 's6', 's8', 's9', 's10', 's13', 's14', 's16', 's17',\
's18', 's19', 'RUL'], axis=1)



#Get Target Feature: RUL
trainY = dataframeTrain.RUL

#Get Dataframe values
trainX = trainX.values
trainX = trainX.astype('float32')


# split into input (X) and output (Y) variables
trainX = trainX[0:45000,]
trainY = trainY[0:45000]

#Trainingsdatensatz Standardisieren
sc = StandardScaler()
sc.fit(trainX)
trainX = sc.transform(trainX)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#Respahe Y to (samples,1)
trainY = numpy.reshape(trainY, (len(trainY),1))


# create and fit the LSTM network
def small_model():
    model = Sequential()

    model.add(LSTM(
    15,
    input_shape=(1,15),
    return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GaussianNoise(0.1))

    model.add(LSTM(
    15,
    return_sequences=False))
    model.add(Dropout(0.1))
    model.add(GaussianNoise(0.1))

    model.add(Dense(1))
    model.add(Activation("linear"))

    model.compile(loss="mse", optimizer=adam)
    return model



model = small_model()

model.fit(trainX, trainY, epochs=500, batch_size=125,validation_split=0.2 ,\
          verbose=1, callbacks=[history, reduce_lr, early_stopping])



"""
#PLot the history of the training
plt.plot(history)
plt.title("History of Epochs")
plt.show()
"""
