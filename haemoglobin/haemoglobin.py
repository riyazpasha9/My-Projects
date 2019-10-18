# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:07:47 2019

@author: riyaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.getcwd())
os.chdir('E:\\Naresh IT\\AI\\Datasets')
data = pd.read_excel('shwetha.xlsx')

data.describe()
data.info()

#renaming the columns
data.rename(columns={'haemoglobin level':'haemoglobin_level'},inplace = True)

female = data[data.gender=='female']
female.describe()
male = data[data.gender=='male']
male.describe()

data1 = female.append(male)

from sklearn.utils import shuffle
df1 = shuffle(data1)

#boxplot of sex and creatinine_value
df1[['age','gender']].boxplot(by = 'gender')

plt.boxplot(df1.age)

plt.scatter(df1.age,df1.haemoglobin_level)
plt.xlabel("age")
plt.ylabel("haemoglobin level")

sns.stripplot(x = "gender", y = "haemoglobin_level", data = df1,jitter = True)

sns.stripplot(x = "gender", y = "age", data = df1,jitter = True)

plt.hist(df1['haemoglobin_level'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel("haemoglobin_level")
plt.ylabel("Frequency")

corre = df1.corr()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
df1.iloc[:, 1] = labelencoder.fit_transform(df1.iloc[:, 1])

#X is the independent varibales also called predictors
X = df1[['age','gender','haemoglobin_level']]

#dependent variable also called outcome
y= df1['risk']

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
roc_auc= metrics.auc(fpr,tpr) # auc= 0.66 avg model

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

#building the decision tree classifier for classification
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier()
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) 

score = model2.score(X_test, y_test)
#score is 93% accuracy
print(score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

pd.crosstab(y_test,y_pred2)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred2)
roc_auc= metrics.auc(fpr,tpr) # AUC =0.93  highest  hence this is the best model with accuracy

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

print(classification_report(y_test, y_pred2))

"""
   precision    recall  f1-score   support

          0       1.00      0.88      0.93        16
          1       0.88      1.00      0.93        14

avg / total       0.94      0.93      0.93        30
"""
model2.feature_importances_

from sklearn.svm import SVC

model_svc = SVC(random_state=42)
model_svc.fit(X_train,y_train)
y_pred4 = model_svc.predict(X_test)
model_svc.score(X_test,y_test) # 86%

# calculating auc
pd.crosstab(y_test,y_pred4)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred4)
roc_auc= metrics.auc(fpr,tpr) # 0.875

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


pd.crosstab(df1.haemoglobin_level,df1.gender).plot(kind='bar')
plt.xlabel('haemoglobin_level')
plt.ylabel('frequency')

plt.hist(df1.haemoglobin_level)


sns.swarmplot(x = "haemoglobin_level", y = "gender", hue = "risk", data =df1)


