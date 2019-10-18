# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:03:45 2019

@author: riyaz
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("E:\\Naresh IT\\project ML\\New folder")
data = pd.read_excel('Modeling.xlsx')
data.drop(columns=['OBS#'],inplace = True)

X = data.iloc[:,:-1]
y = data.iloc[:,30]    

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
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dropout(0.2))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier_keras_fitted=classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

(y_pred>0.5).sum()
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from sklearn.metrics import roc_curve
y_pred_keras = classifier.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras) #0.70

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
