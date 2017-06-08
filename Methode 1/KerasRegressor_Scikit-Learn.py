#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:08:27 2017

@author: entwicklung
Tutorial: http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
"""

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import optimizers
from keras import callbacks
import datetime

# load dataset
dataframeTrain = pandas.DataFrame(pandas.read_csv('train.txt', sep = ' '))
dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))
train = dataframeTrain.values
test = dataframeTest.values
history = callbacks.History()
# split into input (X) and output (Y) variables
X_train = train[0:20000,6:26]
y_train = train[0:20000,27]
X_test = test[53:209,6:26]
y_test = test[53:209,27]



adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.05, nesterov=False)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=10, min_lr=0.0002)
#Define some callbacks



# define the model
def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    
    # Compile model    
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


#Trainingsdatensatz Standardisieren
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)

#Testdatensatz Standardisieren
sc = StandardScaler()
sc.fit(X_test)
X_test = sc.transform(X_test)

#Trainingsmenge & Testmenge
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Modell erzeugen
model = KerasRegressor(build_fn=wider_model, epochs=300, batch_size=300, verbose=0)

for x in range(3):
    #Loop einfügen, um Optimizer mehrere Startpunkte zu geben!?
    his = model.fit(X_train,y_train, callbacks=[history, reduce_lr])
    pred = model.predict(X_test)
    
    #score = model.evaluate(X_test, y_test, verbose=0)
    
    #Get date and time to available to assign the experient
    date = datetime.datetime.now()
    versuch = x + 1
    
    print("Versuch %d am %s"% (versuch, date) )
    #print(pred)
    print("Prediction Max: %d"%pred.max())
    print("Prediction Min: %d"%pred.min())
    
    #Berechne Mean Squared Error
    mseFull = numpy.mean((y_test - pred)**2)
    print("MSE: %d" %mseFull)
    
    #IMport list to array
    x_plot = numpy.asanyarray(his.epoch)
    y_plot = numpy.asanyarray(his.history['loss'])
    plt.plot(x_plot, y_plot)
    plt.title("Loss rate per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
    
    #Time Series Plot
    orgLine = plt.plot(y_test, color='blue', label='Original')
    predLine = plt.plot(pred, color='red', label='Predict')
    plt.legend()
    plt.title('Original vs. Predict Output (Seq: 147)')
    plt.show()


"""
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))

"""

"""
#Zeichne Streudiagramm des echten Preises und des berechneten Preises
print("Streudiagramm")
plt.scatter(y_test, model.predict(X_test))
plt.xlabel("Tatsächliche RUL")
plt.ylabel("Predicted RUL")
plt.title("MLPRegressor (Neural Network)")
"""




exit
