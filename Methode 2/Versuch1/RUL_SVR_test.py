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
from sklearn.preprocessing import StandardScaler
np.set_printoptions(threshold=np.nan)
from sklearn.svm import SVR
import pickle

# load dataset
dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))


#Fügt den Spalten den Feature-Namen hinzu

dataframeTest.columns  = ['index','unit', 'cyles', 'OS1', \
'OS2', 'OS3', 's1', 's2'\
, 's3', 's4', 's5', 's6', 's7', 's8'\
, 's9', 's10', 's11', 's12', 's13', 's14'\
, 's15', 's16', 's17', 's18', 's19', 's20'\
, 's21', 'RUL']



"""
Sequnzlänge bei Unit 150 ist 15

Sequnzlänge bei Unit 84 ist 51

Sequnzlänge bei Unit 99 ist 96

Sequnzlänge bei Unit 73 ist 123

Sequnzlänge bei Unit 67 ist 150

Sequnzlänge bei Unit 189 ist 260
"""

#Speichert nur die 9 wichtigen Sensoren
testX = dataframeTest.drop(['index', 'unit', 'cyles', 'OS1', \
'OS2', 'OS3','s1', 's5', 's6', 's8', 's9', 's10', 's13', 's14', 's16', 's17',\
's18', 's19', 'RUL'], axis=1)

#Get Target Feature: RUL
testY = dataframeTest.RUL

#Get Dataframe values
testX = testX.values
testX = testX.astype('float32')

#Testdatensatz Standardisieren
sc = StandardScaler()
sc.fit(testX)
testX = sc.transform(testX)

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

#Trainingsmenge & Testmenge
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
numInstances = len(testX)


# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

#Anzeigen von Predicted und tatsächlichem Wert
#Test 1
testPredict = loaded_model.predict(testX_0)
testY_orgi = testY_0.as_matrix()
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 150 )')
plt.show()

mse1=np.sum((Error)**2)
mae1=np.sum(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 2
testPredict = loaded_model.predict(testX_1)
testY_orgi = testY_1.as_matrix()
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 84 )')
plt.show()

mse1=np.sum((Error)**2)
mae1=np.sum(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 3
testPredict = loaded_model.predict(testX_2)
testY_orgi = testY_2.as_matrix()
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 99 )')
plt.show()

mse1=np.sum((Error)**2)
mae1=np.sum(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 4
testPredict = loaded_model.predict(testX_3)
testY_orgi = testY_3.as_matrix()
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 73 )')
plt.show()

mse1=np.sum((Error)**2)
mae1=np.sum(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 5
testPredict = loaded_model.predict(testX_4)
testY_orgi = testY_4.as_matrix()
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 67 )')
plt.show()

mse1=np.sum((Error)**2)
mae1=np.sum(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")

#Test 6
testPredict = loaded_model.predict(testX_5)
testY_orgi = testY_5.as_matrix()
Error = testPredict-testY_orgi

#Time Series Plot
plt.plot(testY_orgi, color='blue', label='Original')
plt.plot(testPredict, color='red', label='Predict')
plt.legend()
plt.title('Original vs. Predict Output (Seq: 189 )')
plt.show()

mse1=np.sum((Error)**2)
mae1=np.sum(Error)
print("Mean Squared Error \t %2.3f" % mse1)
print("Mean Absolute Error \t %2.3f" % mae1)
print("---------------------------")