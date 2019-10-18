# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:34:34 2018

@author: riyaz
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir("E:\\Naresh IT\\project ML")
# Importing the dataset
dataset = pd.read_csv('MFG10YearTerminationData.csv')
dataset_edited = dataset.reindex(columns = ['age','length_of_service','city_name','department_name','job_title','store_name','gender_full','termreason_desc','termtype_desc','STATUS_YEAR','BUSINESS_UNIT','STATUS'])

train_set = dataset_edited[dataset.STATUS_YEAR <=2014]
test_set = dataset_edited[dataset.STATUS_YEAR==2015]

#fro train data <=2014
X_train_set = train_set[['age','length_of_service','gender_full','STATUS_YEAR','BUSINESS_UNIT']]

y_train_set = train_set.STATUS

#for test data  ==2015
X_test_set = test_set[['age','length_of_service','gender_full','STATUS_YEAR','BUSINESS_UNIT']]

y_test_set = test_set.STATUS

#dataset_ind_vars = dataset.iloc[:,0:16]
#dataset_business = dataset.iloc[:,17]
#dataset_ind_vars['BUSINESS_UNIT'] = dataset.BUSINESS_UNIT
"""dataset_dep_var = dataset.STATUS
X = dataset_edited.iloc[:, 0:11]
y = dataset_edited.iloc[:, 11]
"""
# Encoding categorical data
#to avoid the dummy varibale trap we take the X = X[:, 1:] all rows from 1st column and rest 
#all column except the 0th index
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train_set_BU = LabelEncoder()
X_train_set.iloc[:, 4] = labelencoder_X_train_set_BU.fit_transform(X_train_set.iloc[:, 4])

labelencoder_X_train_set_GF = LabelEncoder()
X_train_set.iloc[:, 2] = labelencoder_X_train_set_GF.fit_transform(X_train_set.iloc[:, 2])

# for test test enconding
labelencoder_X_test_set_BU = LabelEncoder()
X_test_set.iloc[:, 4] = labelencoder_X_test_set_BU.fit_transform(X_test_set.iloc[:, 4])

labelencoder_X_test_set_GF = LabelEncoder()
X_test_set.iloc[:, 2] = labelencoder_X_test_set_GF.fit_transform(X_test_set.iloc[:, 2])

labelencoder_y_train = LabelEncoder()
y_train_set = labelencoder_y_train.fit_transform(y_train_set)

labelencoder_y_test = LabelEncoder()
y_test_set = labelencoder_y_test.fit_transform(y_test_set)


#X.drop(columns=['city_name','department_name','job_title','store_name','termreason_desc','termtype_desc'],inplace=True)

#X.values
#y.values
# Splitting the dataset into the Training set and Test set

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_set = sc.fit_transform(X_train_set)
X_test_set= sc.transform(X_test_set)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import AdaBoostClassifier
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#number of nodes (neuron) in each layer ,there are 11 independent varibales (11+1)/2 =6
#init = uniform because we need to initialise the weights close to zero but not zero
#input_dim = 11 because we have 11 independent variables.
#**if we have more than 2 oucomes or (categorical varibale) in Dependent variable the output_dim = 3 ,activation = 'softmax'
#if the DV has more than 2 outcomes the optimiser is cross_entropy 
#batch_size is number of observations after which we want to update the weights
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 5))
classifier.add(Dropout(0.2))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier_keras_fitted=classifier.fit(X_train_set, y_train_set, batch_size = 10, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test_set)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_set, y_pred)

from sklearn.metrics import roc_curve
y_pred_keras = classifier.predict(X_test_set).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_set, y_pred_keras)

#AUC value can also be calculated like this.

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


def plot_history(classifier,title='Loss and accuracy (Keras model)'):
    plt.figure(figsize=(15,10))
    plt.subplot(211)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(classifier.history['loss'])
    #plt.plot(network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.subplot(212)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(classifier.history['acc'])
    #plt.plot(network_history.history['val_acc'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
plot_history(classifier_keras_fitted)   
