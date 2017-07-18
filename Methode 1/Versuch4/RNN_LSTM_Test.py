#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:56:23 2017

@author: entwicklung
"""

# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# fix random seed for reproducibility
np.random.seed(7)

print('Loading data...')

dataframeTest = pandas.DataFrame(pandas.read_csv('../PHM008_HERI_testing.txt', sep = ' '))

#Fügt den Spalten den Feature-Namen hinzu
dataframeTest.columns  = ['index1','index2','unit', 'cyles', 'OS1', \
'OS2', 'OS3', 's1', 's2'\
, 's3', 's4', 's5', 's6', 's7', 's8'\
, 's9', 's10', 's11', 's12', 's13', 's14'\
, 's15', 's16', 's17', 's18', 's19', 's20'\
, 's21', 'HERI1', 'HERI2', 'HERI3', 'HERI4', 'HERI5', 'HERI6','RUL']



#Speichert nur die 9 wichtigen Sensoren
testX = dataframeTest.drop(['index1','index2', 'unit', 'cyles', 'OS1', \
'OS2', 'OS3','s1', 's5', 's6', 's8', 's9', 's10', 's13', 's14', 's16', 's17',\
's18', 's19', 'RUL'], axis=1)



#Get Target Feature: RUL
testY = dataframeTest.RUL
testY = testY.values
testY = testY.astype('float32')

#Get Dataframe values
testX = testX.values
testX = testX.astype('float32')


# split into input (X) and output (Y) variables
testX = testX[0:29817] #Sequence 3


#Testdatensatz Standardisieren
sc = StandardScaler()
sc.fit(testX)
testX = sc.transform(testX)

# reshape input to be [samples, time steps, features]
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Testdatensatz auswählen
testX_0 = testX[20624:20638,] #Sequence 150
testY_0 = testY[20624:20638]

testX_1 = testX[11192:11242,] #Sequence 84
testY_1 = testY[11192:11242]

testX_2 = testX[12920:13015,] #Sequence 99
testY_2 = testY[12920:13015]

testX_3 = testX[9884:10006,] #Sequence 73
testY_3 = testY[9884:10006]

testX_4 = testX[9081:9230,] #Sequence 67
testY_4 = testY[9081:9230]

testX_5 = testX[25559:25818,] #Sequence 189
testY_5 = testY[25559:25818]

testX_6 = testX[2869:3083,] #Sequence 21
testY_6 = testY[2869:3083]




# load the model from disk
loaded_model = load_model('finalized_model.h5')

#Anzeigen von Predicted und tatsächlichem Wert
#Test 1
testPredict = loaded_model.predict(testX_0)
testY_orgi = testY_0
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 150 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 2
testPredict = loaded_model.predict(testX_1)
testY_orgi = testY_1
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 84 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 3
testPredict = loaded_model.predict(testX_2)
testY_orgi = testY_2
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 99 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 4
testPredict = loaded_model.predict(testX_3)
testY_orgi = testY_3
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 73 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 5
testPredict = loaded_model.predict(testX_4)
testY_orgi = testY_4
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 67 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 6
testPredict = loaded_model.predict(testX_5)
testY_orgi = testY_5
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 189 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 7
testPredict = loaded_model.predict(testX_6)
testY_orgi = testY_6
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 21 )')
plt.show()

mse1=np.mean((Error)**2)
mae1=np.mean(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")