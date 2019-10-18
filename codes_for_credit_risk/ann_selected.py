# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:03:25 2019

@author: riyaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.getcwd())
os.chdir('E:\\Naresh IT\\project ML\\New folder')
data = pd.read_excel('Modeling.xlsx')
data.drop(columns=['OBS#'],inplace = True)

data.describe()
core=data.corr()
data.DURATION.unique()

#data = data[data.DURATION<=44]

data.rename(columns={'RADIO/TV':'RADIO_TV'},inplace = True)

data1 = data[['CHK_ACCT','DURATION','HISTORY','EMPLOYMENT','MALE_SINGLE','OWN_RES','RADIO_TV',
             'NEW_CAR','USED_CAR','SAV_ACCT','REAL_ESTATE','PROP_UNKN_NONE','OTHER_INSTALL','RENT','RESPONSE']]


plt.boxplot(data.DURATION)
(data1.DURATION>=41).sum()

data1.DURATION.unique()
np.sort(data1.DURATION.unique())

data1.info()
#pd.get_dummies(data,drop_first=True)
data1.CHK_ACCT=data1.CHK_ACCT.astype('str')
data1.HISTORY= data1.HISTORY.astype('str')
data1.SAV_ACCT= data1.SAV_ACCT.astype('str')
data1.EMPLOYMENT= data1.EMPLOYMENT.astype('str')
#data1.PRESENT_RESIDENT= data1.PRESENT_RESIDENT.astype('str')
#data1.JOB= data1.JOB.astype('str')



#separting the categorical and numerical data
data1_cat = data1.select_dtypes(exclude = [np.number])
data1_numerical= data1.select_dtypes(include = [np.number])

data1_cat_dummy=pd.get_dummies(data1_cat)
data1_cat_dummy.columns

data1_cat_dummy.drop(columns=['CHK_ACCT_2'],inplace = True)
data1_cat_dummy.drop(columns=['HISTORY_0'],inplace = True)
data1_cat_dummy.drop(columns=['SAV_ACCT_3'],inplace = True)
data1_cat_dummy.drop(columns=['EMPLOYMENT_0'],inplace = True)
#data1_cat_dummy.drop(columns=['PRESENT_RESIDENT_1'],inplace = True)
#data1_cat_dummy.drop(columns=['AGE_50-60'],inplace = True)
#data1_cat_dummy.drop(columns=['JOB_0'],inplace = True)

data1_final=pd.concat([data1_numerical,data1_cat_dummy],axis = 1)

#data1_final.AMOUNT=np.log(data1_final.AMOUNT)     
#amount = data1_final.AMOUNT                   

#dependent values
y = data1_final.RESPONSE

data1_final.drop(columns=['RESPONSE'],inplace = True)
data1.drop(columns=['RESPONSE'],inplace = True)
#independent variables
X = data1_final

#splitting the data set in test and train part,train has 80% data and test has 20% data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer
#number of nodes (neuron) in each layer ,there are 30 independent varibales (30+1)/2 =30
#init = uniform because we need to initialise the weights close to zero but not zero
#input_dim = 30 because we have 30 independent variables.
#batch_size is number of observations after which we want to update the weights
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 26))
classifier.add(Dropout(0.1))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier_keras_fitted=classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

(y_pred>0.5).sum()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
