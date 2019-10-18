# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:35:06 2019

@author: riyaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.getcwd())
os.chdir('E:\\Naresh IT\\AI\\Datasets')
data = pd.read_excel('Dataset.xlsx')

#renaming the columns
data.rename(columns={'creatinine value':'creatinine_value','creatinine risk':'creatinine_risk'},inplace = True)

#boxplot of sex and creatinine_value
data[['creatinine_value','sex']].boxplot(by = 'sex')


plt.scatter(data.age,data.creatinine_value,color = 'red')
plt.xlabel("Age")
plt.ylabel("creatinine_value")


plt.hist(data['age'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.hist(data['creatinine_value'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel("Creatinine_value")
plt.ylabel("Frequency")

#kernel density plot to smoothen the outliers
sns.distplot(data['creatinine_value'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


data[data.creatinine_value>1.4]

#basic statistics of the data
data.describe()

#looking for the coorelation amonng the variables
data.corr()

data['age'].corr(data['creatinine_value'])

(data.age>=60).sum()

#plot for age and the creatinine_value
pd.crosstab(data[data.age],data[data.creatinine_value]).plot(kind='bar',stacked = True)
#plt.title('plot male and female creatinine values')
plt.xlabel('creatinine_value')
plt.ylabel('frequency')


data[data.creatinine_value<0.7]
data[(data.age>10) &(data.age<=15)]

corre = data.corr()

#X is the independent varibales also called predictors
X = data[['age','creatinine_value']]

#dependent variable also called outcome
y= data['creatinine_risk']

#splitting the data set in test and train part,train has 80% data and test has 20% data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#logistics regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


score = classifier.score(X_test, y_test)
#score is 90% accuracy
print(score)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn import metrics
pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr) # auc= 0.5 this model is not good model

# ploting ROC curve
plt.title('ROC')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

#building thr random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model1 = RandomForestClassifier() # 95% accuracy
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
model1.score(X_test,y_test)

score = model1.score(X_test, y_test)
#score is 96% accuracy
print(score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

pd.crosstab(y_test,y_pred1)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred1)
roc_auc= metrics.auc(fpr,tpr) # auc= 0.75 this is a best model with accuracy

# ploting ROC curve
plt.title('ROC')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

model1.feature_importances_

#building the decision tree classifier for classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier()
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) 

pd.crosstab(y_test,y_pred2)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred2)
roc_auc= metrics.auc(fpr,tpr) # AUC = .71 highest  hence this is the best model with accuracy

# ploting ROC curve
plt.title('ROC')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()


#manually passing the values and predicting the results
test1 = [50,0.7]
test2 = [5,0.2]
test3 = [20,0.6]
test4 = [25,1.7]
         
test1 =np.array(test1).reshape(-1,1).transpose()
test2 =np.array(test2).reshape(-1,1).transpose()
test3 =np.array(test3).reshape(-1,1).transpose()
test4 =np.array(test4).reshape(-1,1).transpose()

model1.predict(test2)
data.describe()

(data.sex=='male').sum()

female = data[data.sex== 'female']
female.describe()
""""
             age  creatinine_value  creatinine_risk
count  80.000000         80.000000         80.00000
mean   53.087500          0.910000          0.05000
std    16.163088          0.191331          0.21932
min    10.000000          0.500000          0.00000
25%    42.750000          0.800000          0.00000
50%    53.500000          0.900000          0.00000
75%    65.000000          1.000000          0.00000
max    88.000000          1.600000          1.00000"""

male = data[data.sex=='male']
male.describe()
"""
         age  creatinine_value  creatinine_risk
count  75.000000         75.000000        75.000000
mean   59.226667          1.328000         0.213333
std    13.893974          1.216504         0.412420
min    25.000000          0.600000         0.000000
25%    49.500000          0.900000         0.000000
50%    61.000000          1.000000         0.000000
75%    69.500000          1.200000         0.000000
max    84.000000          8.300000         1.000000"""

sns.swarmplot(x="sex", y="creatinine_value", data = male)

sns.countplot(male['age'])

sns.stripplot(x = "sex", y = "creatinine_value", data = data,jitter = True)

sns.pointplot(x = "age", y = "creatinine_value", hue = "sex", data =data)

plt.scatter(female.age,female.creatinine_value)
plt.xlabel('age_of female')
plt.ylabel('creatinine_value')

plt.scatter(male.age,male.creatinine_value)
plt.xlabel('age_of male')
plt.ylabel('creatinine_value')
