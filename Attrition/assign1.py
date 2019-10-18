# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:02:41 2018

@author: riyaz
"""
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
os.chdir('E:\\Naresh IT\\AI\\Datasets')
dataset = pd.read_csv('MFG10YearTerminationData.csv')

plt.hist(dataset['age'], color = 'red', edgecolor = 'black',
         bins = int(180/5))

plt.hist(dataset['length_of_service'], color = 'red', edgecolor = 'black',
         bins = int(180/5))

plt.hist(dataset['STATUS_YEAR'], color = 'red', edgecolor = 'black',
         bins = int(180/5))

plt.hist(dataset['store_name'], color = 'red', edgecolor = 'black',
         bins = int(180/5))


dataset[['age','gender_full']].boxplot(by = 'gender_full')

sns.distplot(dataset['age'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(dataset['STATUS_YEAR'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(dataset['length_of_service'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

status_count = dataset[['STATUS_YEAR','STATUS']]

#year = dataset.groupby('STATUS_YEAR').count()

"""df = pd.DataFrame({'STATUS_YEAR':dataset.STATUS_YEAR,'STATUS':dataset.STATUS})
temp = df.groupby(['STATUS_YEAR']).count()
df['STATUS_YEAR'].value_counts()"""

#calculating the count of activa and terminated according to year
active = []
terminated = []
for year in range(2006,2016):
    active.append(sum((dataset.STATUS_YEAR == year) & (dataset.STATUS == 'ACTIVE')))
    terminated.append(sum((dataset.STATUS_YEAR ==year) & (dataset.STATUS == 'TERMINATED')))

status_count = pd.DataFrame({'year':list(range(2006,2016)),'active':active,'terminated':terminated})    

plt.scatter(status_count.year,status_count.active,color = 'red')
plt.scatter(status_count.year,status_count.terminated,color = 'red')

#we are calculating the previous active
#shift(1) is shift by 1 
status_count['previous_active'] = status_count['active'].shift(1)
#rearranginf the columns
status_count = status_count[['year','active','terminated','previous_active']]

#calculating the terminated people percentage
status_count['percent_terminated'] = status_count['terminated']/(status_count['previous_active'])*100

#dropping the NA
status_count.dropna(inplace = True)

status_count['percent_terminated'].mean()

#bar plot of count vs business unit with Staus as fill(active ,terminated)
sns.countplot(x ='STATUS',data=dataset,palette = 'hls')
pd.crosstab(dataset['BUSINESS_UNIT'],dataset['STATUS']).plot(kind = 'bar')

#extracting only terminated employess  and creating as dataframe
#this acts like a filter and separates the status only if its is terminated
terminated_dataset = dataset[dataset.STATUS == 'TERMINATED']

pd.crosstab(terminated_dataset['STATUS_YEAR'],terminated_dataset['termtype_desc']).plot(kind = 'bar')
pd.crosstab(terminated_dataset['STATUS_YEAR'],terminated_dataset['termreason_desc']).plot(kind = 'bar')
pd.crosstab(terminated_dataset['department_name'],terminated_dataset['termreason_desc']).plot(kind = 'bar')

# we are checking the density plot of age and length of service 
#according to status (active or terminated)
#we are actually overlapping the first two graphs and plotting on 0 axis
#we are actually overlapping the second  two graphs and plotting on 1 axis
#this is feature plot as per the R code 
fig, axes = plt.subplots(1, 2, sharey = False,sharex = False)

sns.distplot(dataset.age[dataset.STATUS == 'ACTIVE'], hist=False, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax =axes[0],label = 'active')
axes[0].legend(loc ='best')

sns.distplot(dataset.age[dataset.STATUS == 'TERMINATED'], hist=False, kde=True, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax =axes[0],label = 'terminated')
axes[0].legend(loc ='best')

#we are actually overlapping the second  two graphs and plotting on 1 axis
#this is feature plot as per the R code 
#fig, axes = plt.subplots(1, 2, sharey = False,sharex = False)
  
sns.distplot(dataset.length_of_service[dataset.STATUS == 'ACTIVE'], hist=False, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax =axes[1],label = 'active')
axes[1].legend(loc ='best')

   
sns.distplot(dataset.length_of_service[dataset.STATUS == 'TERMINATED'], hist=False, kde=True, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1},ax =axes[1],label = 'terminated')
axes[1].legend(loc ='best')

#feature plot of age vs status ad box plot
#feature plot of length of service vs status ad box plot
fig, axes = plt.subplots(1, 2, sharey = False,sharex = False)
dataset[['age','STATUS']].boxplot(by = 'STATUS',ax=axes[0])
dataset[['length_of_service','STATUS']].boxplot(by = 'STATUS',ax=axes[1])

"""for runningn the ggplot i have made some changes 
step 1:E:\anacondaNew\Lib\site-packages\ggplot\stats 
 from ->from pandas.lib import Timestamp
change to -> from pandas import Timestamp

step 2: E:\anacondaNew\Lib\site-packages\ggplot\stats
in line number 77
from -> smoothed_data = smoothed_data.sort('x')
change to ->smoothed_data = smoothed_data.sort_values('x')

step 3: E:\anacondaNew\Lib\site-packages\ggplot
in line number 602
from ->fill_levels = self.data[[fillcol_raw, fillcol]].sort(fillcol_raw)[fillcol].unique()
change to ->fill_levels = self.data[[fillcol_raw, fillcol]].sort_values(by=fillcol_raw)[fillcol].unique()
"""

"""from ggplot import *
ggplot(dataset , aes(x = 'factor(BUSINESS_UNIT)' , fill = 'STATUS')) +\
geom_bar(stat ='count',position = 'stack')

ggplot(dataset , aes(x = 'STATUS_YEAR' , y ='factor(termtype_desc)',fill ='termtype_desc')) +\
geom_bar(position ='stack')

ggplot(dataset , aes(x = 'STATUS_YEAR' , y = 'factor(termreason_desc)',fill ='termreason_desc')) +\
geom_bar(position ='stack')

ggplot(dataset , aes(x = 'department_name' , y ='termreason_desc',fill ='termreason_desc')) +\
geom_bar(position ='stack') +\
theme(axis_text_x = element_text(angle = 90, hjust =1))
"""

import random
random.seed(42)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Business_unit = LabelEncoder()
dataset.iloc[:, 17] = labelencoder_Business_unit.fit_transform(dataset.iloc[:, 17])

labelencoder_gender_full = LabelEncoder()
dataset.iloc[:, 12] = labelencoder_gender_full.fit_transform(dataset.iloc[:, 12])

labelencoder_status = LabelEncoder()
dataset.iloc[:, 16] = labelencoder_status.fit_transform(dataset.iloc[:, 16])


train_data = dataset[dataset.STATUS_YEAR <=2014]
test_data = dataset[dataset.STATUS_YEAR ==2015]

#training the data only on the status year <=2014
X_train_var = train_data[['age','length_of_service','gender_full','STATUS_YEAR','BUSINESS_UNIT']]

y_train_var = train_data.STATUS

#testing data is of the year 2015
X_test_var = test_data[['age','length_of_service','gender_full','STATUS_YEAR','BUSINESS_UNIT']]

y_test_var = test_data.STATUS

random.seed(42)


import pydotplus
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image

model = DecisionTreeClassifier()
model.fit(X_train_var,y_train_var)
y_pred = model.predict(X_test_var)

X_test_var = X_test_var.values
y_test_var = y_test_var.values
model.score(X_test_var,y_test_var) #0.95726668010481752

#plot for checking the number of 0's(active) and 1's (termin
l = [sum(y_pred==0),sum(y_pred==1),sum(y_test_var==0),sum(y_test_var==1)]
y_pos = np.arange(len(l))
plt.bar(y_pos,height = l)
plt.xticks(y_pos,['y_pred_0','y_pred_1','y_test_0','y_test_1'])

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model1 = RandomForestClassifier() # 95% accuracy
model1.fit(X_train_var,y_train_var)
y_pred = model1.predict(X_test_var)
model1.score(X_test_var,y_test_var)

model1 = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=4) # 98% accuracy
model1.fit(X_train_var,y_train_var)
y_pred = model1.predict(X_test_var)
model1.score(X_test_var,y_test_var)

# feature importance of model : age and service length is having high importance (90%)
model1.feature_importances_ 


# calculating auc
pd.crosstab(y_test_var,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test_var,y_pred)
roc_auc= metrics.auc(fpr,tpr) # 0.694

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

# using ada boost

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import VotingClassifier

model = DecisionTreeClassifier()
model.fit(X_train_var,y_train_var)
y_pred = model.predict(X_test_var)

model1 = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=4) # 98% accuracy
model1.fit(X_train_var,y_train_var)
y_pred = model1.predict(X_test_var)
model1.score(X_test_var,y_test_var)

model2 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),learning_rate=0.1,n_estimators=50)
model2.fit(X_train_var,y_train_var)
model2.score(X_test_var,y_test_var)

model3 = AdaBoostClassifier(base_estimator=RandomForestClassifier(),learning_rate=0.1,n_estimators=50)
model3.fit(X_train_var,y_train_var)
model3.score(X_test_var,y_test_var)

model_vc = VotingClassifier(estimators=[('a',model),('b',model1),('c',model2),('d',model3)],voting='hard')
model_vc.fit(X_train_var,y_train_var)
model.score(X_test_var,y_test_var)

# ada boost ends here----------------

# SVM

from sklearn.svm import SVC

model_svc = SVC(random_state=42)
model_svc.fit(X_train_var,y_train_var)
model_svc.score(X_test_var,y_test_var) # 98%

# pipeline for logit, SVM, dec tree, random forest.


