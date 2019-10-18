# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:33:29 2018

@author: riyaz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
os.chdir('E:\\Naresh IT\\AI\\Datasets')
# Importing the dataset
#X['BusinessUnit'] = dataset.BusinessUnit for adding the last column for analysis
dataset = pd.read_csv('MFGEmployees4.csv')

X = dataset.iloc[:,0:13]
X.drop(columns=['Surname','GivenName','EmployeeNumber','AbsentHours'],inplace=True)
y = dataset.iloc[:, 11]

plt.hist(dataset.Age,bins = int(180/5),histtype = 'bar')

plt.hist(dataset['AbsentHours'], color = 'red', edgecolor = 'black',
         bins = int(180/5))

plt.hist(dataset['Age'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))

plt.hist(dataset['LengthService'], color = 'pink', edgecolor = 'black',
         bins = int(180/5))



sns.distplot(dataset['Age'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(dataset['LengthService'], hist=True, kde=True, 
             bins=int(180/5), color = 'green',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(dataset['AbsentHours'], hist=True, kde=True, 
             bins=int(180/5), color = 'orange',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


sns.distplot(dataset['Age'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


(dataset['Age']>=20).sum()

plt.boxplot(dataset.Age,showmeans=True)
plt.boxplot(dataset.LengthService,showmeans=True)
plt.boxplot(dataset.AbsentHours,showmeans=True)

dataset1 = dataset

#removing outliers of age 
AgeSubset = (dataset['Age']>=18).values
dataset1 = dataset[AgeSubset]

#again in dataset1 removig the outliers and updating the dataset1
AgeSubset = (dataset1['Age']<=65).values
dataset1 = dataset1[AgeSubset]

plt.boxplot(dataset1.Age,showmeans=True)

#absent ratio calculation
#2080 => is 52 weeks in one year *per week 5 days working *8 hours per day
#dropping the absentHours because we have calculate the absent rate.
AbsentRate = (dataset1['AbsentHours']/2080)*100
dataset1['AbsentRate'] = AbsentRate
dataset1.drop(columns=['AbsentHours'],inplace = True)


sns.distplot(dataset1['AbsentRate'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

plt.boxplot(dataset1.AbsentRate,showmeans=True)

#Numrical varibale analysis

#plot in dataset1 by absent rate vs age
plt.scatter(dataset1.Age,dataset1.AbsentRate,color = 'red')
plt.title('absentRate vs Age')
plt.xlabel('Age')
plt.ylabel('AbsentRate')
plt.show()

#finding the correlation between age and absent rate
dataset1['Age'].corr(dataset1['AbsentRate'])

#plot in dataset1 by lenghtservice vs age
plt.scatter(dataset1.Age,dataset1.LengthService,color = 'red')
plt.title('LengthService vs Age')
plt.xlabel('Age')
plt.ylabel('LengthService')
plt.show()

#finding the correlation between age and length service
dataset1['Age'].corr(dataset1['LengthService'])

#categorical
#boxplot of absent rate and gender
dataset1[['AbsentRate','Gender']].boxplot(by = 'Gender')
dataset1[['Age','Gender']].boxplot(by = 'Gender')

dataset1.drop(columns=['Surname','GivenName','EmployeeNumber'],inplace=True)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_dataset1 = LabelEncoder()
dataset1.iloc[:, 0] = labelencoder_dataset1.fit_transform(dataset1.iloc[:, 0])

X = dataset1.iloc[:,0:9]
y = dataset1.iloc[:, 9] 

X = pd.get_dummies(X[['City','DepartmentName','JobTitle','StoreLocation','Division','BusinessUnit']],drop_first=True)   
X['Age'] = dataset1.Age
X['LengthService']  = dataset1.LengthService
X['Gender'] = dataset1.Gender 


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test)
'''

from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(criterion = 'mse',random_state = 0,max_depth = 3,min_samples_leaf = 5)
regressor1.fit(X,y)
#y_pred1 = regressor.predict()
#regressor1.score(X_test,y_test)

kfold = KFold(n_splits = 20,random_state =0)
result = cross_val_score(regressor1,X,y,cv=kfold)
result.mean()
"""import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
regressor = ols('dataset1.AbsentRate ~ dataset1.Age + dataset1.LengthService+ dataset1.Division + dataset1.Age*dataset1.Division'  ,data=dataset1).fit()
regressor.params
z = regressor.conf_int()
regressor.summary()

aov_table = sm.stats.anova_lm(regressor, typ=2)
aov_table

sns.pairplot(dataset1, x_vars=['Age','LengthService','Gender'], y_vars='AbsentRate', size=5, aspect=0.7, kind='reg')
"""
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor()
regressor2.fit(X,y)
regressor2.feature_importances_

#calculating the impotance features on random forest and sorting them
feature_importances = pd.DataFrame(regressor2.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


kfold = KFold(n_splits = 20,random_state =0)
result1 = cross_val_score(regressor2,X,y,cv=kfold)
result1.mean()

#only on age
X_age['Gender']= dataset1.iloc[:,0]
y_abrate = dataset1.AbsentRate


from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor()
regressor2.fit(X_age,y_abrate)

kfold = KFold(n_splits = 20,random_state =0)
result1 = cross_val_score(regressor2,X_age,y_abrate,cv=kfold)
result1.mean()