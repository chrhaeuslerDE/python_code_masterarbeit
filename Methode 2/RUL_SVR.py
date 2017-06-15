#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 14:07:36 2017

@author: entwicklung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:14:47 2017

@author: entwicklung"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
np.set_printoptions(threshold=np.nan)
from sklearn.svm import SVR

# load dataset
dataframeTrain = pandas.DataFrame(pandas.read_csv('train.txt', sep = ' '))
dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))


#Fügt den Spalten den Feature-Namen hinzu
dataframeTrain.columns  = ['index','unit', 'cyles', 'OS1', \
'OS2', 'OS3', 's1', 's2'\
, 's3', 's4', 's5', 's6', 's7', 's8'\
, 's9', 's10', 's11', 's12', 's13', 's14'\
, 's15', 's16', 's17', 's18', 's19', 's20'\
, 's21', 'RUL']

dataframeTest.columns  = ['index','unit', 'cyles', 'OS1', \
'OS2', 'OS3', 's1', 's2'\
, 's3', 's4', 's5', 's6', 's7', 's8'\
, 's9', 's10', 's11', 's12', 's13', 's14'\
, 's15', 's16', 's17', 's18', 's19', 's20'\
, 's21', 'RUL']

#Speichert nur die 9 wichtigen Sensoren
trainX = dataframeTrain.drop(['index', 'unit', 'cyles', 'OS1', \
'OS2', 'OS3','s1', 's5', 's6', 's8', 's9', 's10', 's13', 's14', 's16', 's17',\
's18', 's19', 'RUL'], axis=1)

testX = dataframeTest.drop(['index', 'unit', 'cyles', 'OS1', \
'OS2', 'OS3','s1', 's5', 's6', 's8', 's9', 's10', 's13', 's14', 's16', 's17',\
's18', 's19', 'RUL'], axis=1)

#Get Target Feature: RUL
trainY = dataframeTrain.RUL
testY = dataframeTest.RUL


#Get Dataframe values
trainX = trainX.values
trainX = trainX.astype('float32')
testX = testX.values
testX = testX.astype('float32')

# split into input (X) and output (Y) variables
trainX = trainX[0:15000,]
trainY = trainY[0:15000]

testX = testX[53:209,] #Sequence 2
testY = testY[53:209]

testY_orginal = testY

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
#Trainingsdatensatz Standardisieren
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)

#Testdatensatz Standardisieren
sc = StandardScaler()
sc.fit(X_test)
X_test = sc.transform(X_test)
"""
#Trainingsmenge & Testmenge
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
numInstances = len(testX)

#Create machine learning algorithmen
svr = SVR(kernel='rbf', gamma='auto', coef0=0.0, tol=0.001, \
          C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=True,\
          max_iter=-1)
svr.fit(trainX, trainY)

#Anzeigen von Predicted und tatsächlichem Wert
testPredict = svr.predict(testX)

# invert predictions
testPredict = scalerTestY.inverse_transform(testPredict)

"""
#Zeichne Streudiagramm des echten Preises und des berechneten Preises
print("Streudiagramm")
plt.scatter(y_test, predictedOutput)
plt.xlabel("Tatsächliche RUL")
plt.ylabel("Predicted RUL")
plt.title("Support Vector Regression")
"""



Error = testPredict-testY_orginal

#Time Series Plot
orgLine = plt.plot(dataframeTest.RUL[53:209].values, color='blue', label='Original')
predLine = plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: )')
plt.show()


mse1=1.0/numInstances*np.sum((Error)**2)
mad1=1.0/numInstances*np.sum(np.abs(Error))
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Difference \t %2.3f" % mad1)
print("---------------------------")