# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 18:11:54 2019

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

data1.AGE=data.AGE.map({67:'50-60',22:'20-30',49:'40-50',45:'40-50',53:'50-60',35:'30-40',61:'50-60',28:'20-30',25:'20-30',24:'20-30',
              60:'50-60',32:'30-40',44:'40-50',31:'30-40',48:'40-50',26:'20-30',36:'30-40',39:'30-40',42:'40-50',34:'30-40',
              63:'50-60',27:'20-30',30:'20-30',57:'50-60',33:'30-40',37:'30-40',58:'50-60',23:'20-30',29:'20-30',52:'50-60',
              50:'40-50',46:'40-50',51:'50-60',41:'40-50',40:'30-40',66:'50-60',47:'40-50',56:'50-60',54:'50-60',20:'20-30',
              21:'20-30',38:'30-40',70:'50-60',65:'50-60',74:'50-60',68:'50-60',43:'40-50',55:'50-60',64:'50-60',75:'50-60',
              19:'20-30',62:'50-60',59:'50-60'})

data1.AGE.unique()
data1.AGE.value_counts()

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
y = data1_final.RESPONSE.values

data1_final.drop(columns=['RESPONSE'],inplace = True)

#independent variables
X = data1_final.values

#splitting the data set in test and train part,train has 80% data and test has 20% data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,)
classifier.fit(X_train, y_train)
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)

score = classifier.score(X_test, y_test)
print(score) #76%

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3))

#---------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier(max_depth = 5,random_state =42)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) #67.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

#------------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_model = AdaBoostClassifier(base_estimator =DecisionTreeClassifier(),learning_rate = 0.1, n_estimators = 10 )
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ada)
ada_model.score(X_test,y_test)

from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier(learning_rate = 0.1,n_estimators = 100, subsample = 0.5,max_features = 0.15)
gbc_model.fit(X_train, y_train)

y_pred_gbc = gbc_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_gbc)

gbc_model.score(X_test,y_test) #76%
cm = confusion_matrix(y_test, y_pred_gbc)

#---------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model1 = RandomForestClassifier(random_state = 0) 
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
model1.score(X_test,y_test) #74%

score = model1.score(X_test, y_test)
print(score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

features=model1.feature_importances_

#----------------------------------------------------------------------------
#SVC model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(C=1000,kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #75%
print(classification_report(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
accuracies.mean() #76%
accuracies.std()

from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100,1000], 'kernel': ['linear']},
              {'C': [1, 10, 100,1000], 'kernel': ['rbf'], 'gamma': [0.01,0.001]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 20)
                           #n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#----------------------------------------------------------------
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(learning_rate =0.1,
 n_estimators=50,
 max_depth=3,
 min_child_weight=1,
 gamma=0.1,
 subsample=1,
 colsample_bytree=1,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #75.5%