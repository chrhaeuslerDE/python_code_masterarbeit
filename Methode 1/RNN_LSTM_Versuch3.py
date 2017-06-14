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
import math, pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
print('Loading data...')
dataframeTrain = pandas.DataFrame(pandas.read_csv('train.txt', sep = ' '))
dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))
datasetTr = dataframeTrain.values
datasetTr = datasetTr.astype('float32')
datasetTe = dataframeTest.values
datasetTe = datasetTe.astype('float32')

"""
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
"""
# split into input (X) and output (Y) variables
trainX = datasetTr[0:45000,6:26]
trainY = datasetTr[0:45000,27]
testX = datasetTe[53:209,6:26] #Sequence 2
testY = datasetTe[53:209,27]

# normalize the dataset
scalerTrainX = MinMaxScaler(feature_range=(0, 1))
trainX = scalerTrainX.fit_transform(trainX)

scalerTrainY = MinMaxScaler(feature_range=(0, 1))
trainY = scalerTrainY.fit_transform(trainY)

scalerTestX = MinMaxScaler(feature_range=(0, 1))
testX = scalerTestX.fit_transform(testX)

scalerTestY = MinMaxScaler(feature_range=(0, 1))
testY = scalerTestY.fit_transform(testY)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        layers[0],
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

    model.compile(loss="mse", optimizer="rmsprop")
    return model


model = build_model([1, 50, 100, 1])

model.fit(trainX, trainY, epochs=1000, batch_size=500,validation_split=0.05 ,\
          verbose=1)

# make predictions
testPredict = model.predict(testX)

# invert predictions
testPredict = scalerTestY.inverse_transform(testPredict)
testY = scalerTestY.inverse_transform([testY])
"""
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
                 
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
"""
# plot baseline and predictions
plt.plot(testPredict)
testY = datasetTe[53:209,27]
plt.plot(testY)
plt.show()
