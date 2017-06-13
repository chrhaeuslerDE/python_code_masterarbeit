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
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

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
model = Sequential()
model.add(LSTM(4, input_shape=(1, 20)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=100, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scalerTrainY.inverse_transform(trainPredict)
trainY = scalerTrainY.inverse_transform([trainY])
testPredict = scalerTestY.inverse_transform(testPredict)
testY = scalerTestY.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

"""
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
                 
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
"""