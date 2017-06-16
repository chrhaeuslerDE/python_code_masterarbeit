#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 08:37:36 2017

@author: entwicklung
@Beschreibung: Durch einen Plot ist erkenntlich wie die Verteilung der
Sensorwerte aussieht. Damit kann die Auswahl wichtiger Features vorgenommen
werden. Durch ändern der Variable 'sensor' kann die Werteverteilung aller 
Units von 1-218 eingesehen werden.
"""

import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
np.set_printoptions(threshold=np.nan)
from sklearn.svm import SVR

#Größe des Plots änderbar
fig = plt.gcf()
fig.set_size_inches(9.5, 6.5)

# load dataset
dataframeTrain = pandas.DataFrame(pandas.read_csv('train.txt', sep = ' '))
#dataframeTest = pandas.DataFrame(pandas.read_csv('test.txt', sep = ' '))

dataframeTrain.columns  = ['index','unit', 'cyles', 'OS1', \
'OS2', 'OS3', 's1', 's2'\
, 's3', 's4', 's5', 's6', 's7', 's8'\
, 's9', 's10', 's11', 's12', 's13', 's14'\
, 's15', 's16', 's17', 's18', 's19', 's20'\
, 's21', 'RUL']


columns = dataframeTrain.columns
count = 0

for x in range(1,218):
    df = dataframeTrain[dataframeTrain.unit == x]
    samples = df.values
    count = count + len(samples)
    
    """

    for x in range (6,28):
        
        sensor_values = df.iloc[:,x]
        sensor_name = columns[x]
       
        
        sensor_values = samples[:,x]
        sensor_name = columns[x]
        
        
        plt.title("Sensorwerte %s"%sensor_name)
        plt.plot(sensor_values, '+')
        
    
    """
    sensor=6
    sensor_values = samples[:,sensor]
    sensor_name = columns[sensor]    
    plt.title("Sensorwerte von %s"%sensor_name)
    plt.plot(sensor_values, 'b+')