#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Mon May 15 13:18:16 2017

@author: entwicklung"""

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.datasets import imdb
import pandas
import matplotlib.pyplot as plt
from keras import optimizers
from keras import callbacks
import numpy
from numpy import newaxis
import keras.regularizers as regularizers
from sklearn.preprocessing import StandardScaler



tsteps = 20
batch_size = 25
epochs = 25

history = callbacks.History()

print('Loading data...')

dataframeTrain = pandas.DataFrame(pandas.read_csv('train.txt', sep = ' '))
dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))
train = dataframeTrain.values
test = dataframeTest.values

# split into input (X) and output (Y) variables
X_train = train[0:10000,6:26]
y_train = train[0:10000,27]
X_test = test[53:209,6:26] #Sequence 2
y_test = test[53:209,27]

#Trainingsdatensatz Standardisieren
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)

#Testdatensatz Standardisieren
sc = StandardScaler()
sc.fit(X_test)
X_test = sc.transform(X_test)

#Reshape from 2D to 3D Array
X_train = X_train[:, :, None]
#y_train = y_train[:, :, newaxis]
X_test = X_test[:, :, None]
#y_test = y_test[:, :, newaxis]


adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                       decay=0.0)
sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.05, nesterov=False)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.0002)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, 
                                         mode='auto')

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
print('Pad sequences (samples x time)')
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)


def lstm():
    model = Sequential()
    model.add(LSTM(50,
                   input_shape=(tsteps, 1),
                   batch_size=batch_size,
                   return_sequences=True,
                   stateful=True, activation='relu'))
    model.add(LSTM(50,
                   return_sequences=False,
                   stateful=True, activation='relu'))
    model.add(LSTM(1, activation='linear'))
    model.compile(loss='mse', optimizer=adam)  
    return model

print('Build model...')
model = lstm()

print('Train...')
his = model.fit(X_train, y_train, batch_size=batch_size, epochs=50, verbose=1, 
      callbacks=[history, reduce_lr, early_stopping], validation_split=0.0, 
      validation_data=None, class_weight=None, 
      sample_weight=None, initial_epoch=0, shuffle=True)

score = model.evaluate(X_test, y_test,
                            batch_size=batch_size)

pred = model.predict(X_test)


#Berechne Mean Squared Error
mseFull = numpy.mean((y_test - pred)**2)
print("MSE: %d" %mseFull)
#Time Series Plot
orgLine = plt.plot(y_test, color='blue', label='Original')
predLine = plt.plot(pred, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 2)')
plt.show()

#IMport list to array
x_plot = numpy.asanyarray(his.epoch)
y_plot = numpy.asanyarray(his.history['loss'])
plt.plot(x_plot, y_plot)
plt.title("Loss rate per epoch")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()




