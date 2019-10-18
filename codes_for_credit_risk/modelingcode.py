# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:20:36 2019

@author: riyaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.getcwd())
os.chdir('E:\\Naresh IT\\project ML\\Credits_risk_analysis\\codes_for_credit_risk')
data = pd.read_excel('Modeling.xlsx')
data.drop(columns=['OBS#'],inplace = True)

basic_stat_of_data=data.describe()
core=data.corr()

data.rename(columns={'RADIO/TV':'RADIO_TV'},inplace = True)

# Exploratory Data Analysis (EDA) #######################

sns.stripplot(x = "DURATION", y = "AMOUNT", data =data,jitter = True)
sns.stripplot(x = "MALE_DIV", y = "AMOUNT", data = data,jitter = True)
sns.stripplot(x = "MALE_SINGLE", y = "AMOUNT", data = data,jitter = True)
sns.stripplot(x = "MALE_MAR_or_WID", y = "AMOUNT", data = data,jitter = True)

sns.swarmplot(x = "DURATION", y = "AMOUNT", hue = "MALE_DIV", data =data)
sns.swarmplot(x = "DURATION", y = "AMOUNT", hue = "MALE_SINGLE", data =data)
sns.swarmplot(x = "DURATION", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)

pd.crosstab(data.DURATION,data.RETRAINING).plot(kind='bar')
plt.xlabel('DURATION')
plt.ylabel('frequency')

sns.pairplot(data, hue='JOBS', size=2.5);

plt.scatter(data.AMOUNT,data.DURATION,color= 'red')

sns.regplot(x=data["AMOUNT"], y=data["DURATION"],fit_reg=True)

data[['AGE','MALE_DIV']].boxplot(by = 'MALE_DIV')
data[['AGE','MALE_MAR_or_WID']].boxplot(by = 'MALE_MAR_or_WID')
data[['AMOUNT','DURATION']].boxplot(by = 'DURATION')

sns.swarmplot(x = "NEW_CAR", y = "AMOUNT", hue = "MALE_DIV", data =data)
sns.swarmplot(x = "USED_CAR", y = "AMOUNT", hue = "MALE_DIV", data =data)
sns.swarmplot(x = "EDUCATION", y = "AMOUNT", hue = "MALE_DIV", data =data)
sns.swarmplot(x = "RADIO_TV", y = "AMOUNT", hue = "MALE_DIV", data =data)
sns.swarmplot(x = "RETRAINING", y = "AMOUNT", hue = "MALE_DIV", data =data)
sns.swarmplot(x = "FURNITURE", y = "AMOUNT", hue = "MALE_DIV", data =data)

sns.swarmplot(x = "NEW_CAR", y = "AMOUNT", hue = "MALE_SINGLE", data =data)
sns.swarmplot(x = "USED_CAR", y = "AMOUNT", hue = "MALE_SINGLE", data =data)
sns.swarmplot(x = "EDUCATION", y = "AMOUNT", hue = "MALE_SINGLE", data =data)
sns.swarmplot(x = "RADIO_TV", y = "AMOUNT", hue = "MALE_SINGLE", data =data)
sns.swarmplot(x = "RETRAINING", y = "AMOUNT", hue = "MALE_SINGLE", data =data)
sns.swarmplot(x = "FURNITURE", y = "AMOUNT", hue = "MALE_SINGLE", data =data)

sns.swarmplot(x = "NEW_CAR", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)
sns.swarmplot(x = "USED_CAR", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)
sns.swarmplot(x = "EDUCATION", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)
sns.swarmplot(x = "RADIO_TV", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)
sns.swarmplot(x = "RETRAINING", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)
sns.swarmplot(x = "FURNITURE", y = "AMOUNT", hue = "MALE_MAR_or_WID", data =data)
    
sns.barplot(x="REAL_ESTATE", y="AMOUNT", data=data)
sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", data=data)
sns.barplot(x="OTHER_INSTALL", y="AMOUNT", data=data)

sns.barplot(x="REAL_ESTATE", y="AMOUNT", hue="NEW_CAR", data=data)
sns.barplot(x="REAL_ESTATE", y="AMOUNT", hue="USED_CAR", data=data)
sns.barplot(x="REAL_ESTATE", y="AMOUNT", hue="EDUCATION", data=data)
sns.barplot(x="REAL_ESTATE", y="AMOUNT", hue="FURNITURE", data=data)
sns.barplot(x="REAL_ESTATE", y="AMOUNT", hue="RADIO_TV", data=data)
sns.barplot(x="REAL_ESTATE", y="AMOUNT", hue="RETRAINING", data=data)

sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", hue="NEW_CAR", data=data)
sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", hue="USED_CAR", data=data)
sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", hue="EDUCATION", data=data)
sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", hue="FURNITURE", data=data)
sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", hue="RADIO_TV", data=data)
sns.barplot(x="PROP_UNKN_NONE", y="AMOUNT", hue="RETRAINING", data=data)

sns.countplot(x="PROP_UNKN_NONE", hue="NEW_CAR", data=data)
sns.countplot(x="PROP_UNKN_NONE", hue="USED_CAR", data=data)
sns.countplot(x="PROP_UNKN_NONE", hue="EDUCATION", data=data)
sns.countplot(x="PROP_UNKN_NONE", hue="FURNITURE", data=data)
sns.countplot(x="PROP_UNKN_NONE", hue="RADIO_TV", data=data)
sns.countplot(x="PROP_UNKN_NONE", hue="RETRAINING", data=data)

sns.countplot(x="REAL_ESTATE", hue="NEW_CAR", data=data)
sns.countplot(x="REAL_ESTATE", hue="USED_CAR", data=data)
sns.countplot(x="REAL_ESTATE", hue="EDUCATION", data=data)
sns.countplot(x="REAL_ESTATE", hue="FURNITURE", data=data)
sns.countplot(x="REAL_ESTATE", hue="RADIO_TV", data=data)
sns.countplot(x="REAL_ESTATE", hue="RETRAINING", data=data)

sns.countplot(x="OTHER_INSTALL", hue="NEW_CAR", data=data)
sns.countplot(x="OTHER_INSTALL", hue="USED_CAR", data=data)
sns.countplot(x="OTHER_INSTALL", hue="EDUCATION", data=data)
sns.countplot(x="OTHER_INSTALL", hue="FURNITURE", data=data)
sns.countplot(x="OTHER_INSTALL", hue="RADIO_TV", data=data)
sns.countplot(x="OTHER_INSTALL", hue="RETRAINING", data=data)

sns.countplot(x="RENT", hue="NEW_CAR", data=data)
sns.countplot(x="RENT", hue="USED_CAR", data=data)
sns.countplot(x="RENT", hue="EDUCATION", data=data)
sns.countplot(x="RENT", hue="FURNITURE", data=data)
sns.countplot(x="RENT", hue="RADIO_TV", data=data)
sns.countplot(x="RENT", hue="RETRAINING", data=data)

sns.countplot(x="OWN_RES", hue="NEW_CAR", data=data)
sns.countplot(x="OWN_RES", hue="USED_CAR", data=data)
sns.countplot(x="OWN_RES", hue="EDUCATION", data=data)
sns.countplot(x="OWN_RES", hue="FURNITURE", data=data)
sns.countplot(x="OWN_RES", hue="RADIO_TV", data=data)
sns.countplot(x="OWN_RES", hue="RETRAINING", data=data)

sns.countplot(x="JOB", hue="NEW_CAR", data=data)
sns.countplot(x="JOB", hue="USED_CAR", data=data)
sns.countplot(x="JOB", hue="EDUCATION", data=data)
sns.countplot(x="JOB", hue="FURNITURE", data=data)
sns.countplot(x="JOB", hue="RADIO_TV", data=data)
sns.countplot(x="JOB", hue="RETRAINING", data=data)

sns.countplot(x="NUM_CREDITS", hue="NEW_CAR", data=data)
sns.countplot(x="NUM_CREDITS", hue="USED_CAR", data=data)
sns.countplot(x="NUM_CREDITS", hue="EDUCATION", data=data)
sns.countplot(x="NUM_CREDITS", hue="FURNITURE", data=data)
sns.countplot(x="NUM_CREDITS", hue="RADIO_TV", data=data)
sns.countplot(x="NUM_CREDITS", hue="RETRAINING", data=data)

sns.countplot(x="NUM_DEPENDENTS", hue="NEW_CAR", data=data)
sns.countplot(x="NUM_DEPENDENTS", hue="USED_CAR", data=data)
sns.countplot(x="NUM_DEPENDENTS", hue="EDUCATION", data=data)
sns.countplot(x="NUM_DEPENDENTS", hue="FURNITURE", data=data)
sns.countplot(x="NUM_DEPENDENTS", hue="RADIO_TV", data=data)
sns.countplot(x="NUM_DEPENDENTS", hue="RETRAINING", data=data)

sns.countplot(x="FOREIGN", hue="NEW_CAR", data=data)
sns.countplot(x="FOREIGN", hue="USED_CAR", data=data)
sns.countplot(x="FOREIGN", hue="EDUCATION", data=data)
sns.countplot(x="FOREIGN", hue="FURNITURE", data=data)
sns.countplot(x="FOREIGN", hue="RADIO_TV", data=data)
sns.countplot(x="FOREIGN", hue="RETRAINING", data=data)

sns.countplot(x="MALE_DIV", hue="RESPONSE", data=data)
sns.countplot(x="MALE_SINGLE", hue="RESPONSE", data=data)
sns.countplot(x="MALE_MAR_or_WID", hue="RESPONSE", data=data)
sns.countplot(x="CO-APPLICANT", hue="RESPONSE", data=data)
sns.countplot(x="GUARANTOR", hue="RESPONSE", data=data)
sns.countplot(x="PRESENT_RESIDENT", hue="RESPONSE", data=data)

sns.countplot(x="REAL_ESTATE", hue="RESPONSE", data=data)
sns.countplot(x="PROP_UNKN_NONE", hue="RESPONSE", data=data)
sns.countplot(x="OTHER_INSTALL", hue="RESPONSE", data=data)
sns.countplot(x="RENT", hue="RESPONSE", data=data)
sns.countplot(x="OWN_RES", hue="RESPONSE", data=data)
sns.countplot(x="JOB", hue="RESPONSE", data=data)
sns.countplot(x="NUM_CREDITS", hue="RESPONSE", data=data)
sns.countplot(x="INSTALL_RATE", hue="RESPONSE", data=data)
sns.countplot(x="EMPLOYMENT", hue="RESPONSE", data=data)
sns.countplot(x="SAV_ACCT", hue="RESPONSE", data=data)
sns.countplot(x="NUM_DEPENDENTS", hue="RESPONSE", data=data)
sns.countplot(x="FOREIGN", hue="RESPONSE", data=data)
sns.countplot(x="AGE", hue="RESPONSE", data=data)
sns.countplot(x="HISTORY", hue="RESPONSE", data=data)
sns.countplot(x="CHK_ACCT", hue="RESPONSE", data=data)
sns.countplot(x="AGE", hue="RESPONSE", data=data)


sns.barplot(x = "AGE", y = "RESPONSE", hue = "MALE_SINGLE", data =data)

data[['AGE','RESPONSE']].boxplot(by = 'RESPONSE')

plt.hist(data['AMOUNT'], color = 'red', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel("amount")
plt.ylabel("frequency")
plt.title("histogram_of_amount")

plt.hist(data['MALE_SINGLE'], color = 'red', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel("male_single")
plt.ylabel("frequency")
plt.title("histogram_of_male_single")

sns.distplot(data['AMOUNT'], hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
#EDA ends here------------------------------------------------------------------------------------------

#building the models without data preparation
X = data.iloc[:,:-1]
X.drop(columns=['OBS#'],inplace = True)

y = data.iloc[:,31]            

#splitting the data set in test and train part,train has 80% data and test has 20% data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model1 = RandomForestClassifier(max_depth = 10,random_state = 0,) 
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
model1.score(X_test,y_test)

score = model1.score(X_test, y_test)
print(score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

features=model1.feature_importances_

pd.crosstab(y_test,y_pred1)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred1)
roc_auc= metrics.auc(fpr,tpr) # auc= 0.65 avg model

#-----------------------------------------------------------------------------------------
#decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier(max_depth = 7,random_state =42)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) 

pd.crosstab(y_test,y_pred2)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred2)
roc_auc= metrics.auc(fpr,tpr) #0.62

#--------------------------------------------------------------------------------------
#logistics regression model

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,)
classifier.fit(X_train, y_train)
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)


score = classifier.score(X_test, y_test)
print(score) #75%

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3))

from sklearn import metrics
pd.crosstab(y_test,y_pred3)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred3)
roc_auc= metrics.auc(fpr,tpr) # auc= 0.70 

#-----------------------------------------------------------------------------
#SVC model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)    
                
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #75%
print(classification_report(y_test, y_pred))

pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr) #0.67

#------------------------------------------------------------------------------
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.score(X_test,y_test) #70%

pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr) #0.66

#-------------------------------------------------------------------------------
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

classifier.score(X_test,y_test) #73%

pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr) #0.6

#------------------------------------------------------------------------------------
#Data preparation
#

#checking for the normal distrbution of amount
plt.hist(np.log((data.AMOUNT)))

np.log((data.AMOUNT)).skew() #0.129

#I have made the age groups 20-30,30-40,40-50,50-60 and above sixty i have put those values in 50-60 group
data.AGE.unique()
data.AGE=data.AGE.map({67:'50-60',22:'20-30',49:'40-50',45:'40-50',53:'50-60',35:'30-40',61:'50-60',28:'20-30',25:'20-30',24:'20-30',
              60:'50-60',32:'30-40',44:'40-50',31:'30-40',48:'40-50',26:'20-30',36:'30-40',39:'30-40',42:'40-50',34:'30-40',
              63:'50-60',27:'20-30',30:'20-30',57:'50-60',33:'30-40',37:'30-40',58:'50-60',23:'20-30',29:'20-30',52:'50-60',
              50:'40-50',46:'40-50',51:'50-60',41:'40-50',40:'30-40',66:'50-60',47:'40-50',56:'50-60',54:'50-60',20:'20-30',
              21:'20-30',38:'30-40',70:'50-60',65:'50-60',74:'50-60',68:'50-60',43:'40-50',55:'50-60',64:'50-60',75:'50-60',
              19:'20-30',62:'50-60',59:'50-60'})

data.AGE.value_counts()

plt.boxplot(data.DURATION)
(data.DURATION>=41).sum()

data.DURATION.unique()
np.sort(data.DURATION.unique())

data.info()
#as the type was int64 so iam converting them into str type for dummy creating dummy variables
data.CHK_ACCT=data.CHK_ACCT.astype('str')
data.HISTORY= data.HISTORY.astype('str')
data.SAV_ACCT= data.SAV_ACCT.astype('str')
data.EMPLOYMENT= data.EMPLOYMENT.astype('str')
data.PRESENT_RESIDENT= data.PRESENT_RESIDENT.astype('str')
data.JOB= data.JOB.astype('str')

#separting the categorical and numerical data
data_cat = data.select_dtypes(exclude = [np.number])
data_numerical= data.select_dtypes(include = [np.number])

#creating the dummy variables
data_cat_dummy=pd.get_dummies(data_cat)
data_cat_dummy.columns

data_cat_dummy.drop(columns=['CHK_ACCT_2'],inplace = True)
data_cat_dummy.drop(columns=['HISTORY_0'],inplace = True)
data_cat_dummy.drop(columns=['SAV_ACCT_3'],inplace = True)
data_cat_dummy.drop(columns=['EMPLOYMENT_0'],inplace = True)
data_cat_dummy.drop(columns=['PRESENT_RESIDENT_1'],inplace = True)
data_cat_dummy.drop(columns=['AGE_50-60'],inplace = True)
data_cat_dummy.drop(columns=['JOB_0'],inplace = True)

data_final=pd.concat([data_numerical,data_cat_dummy],axis = 1)
data_final.to_csv('E:\\Naresh IT\\project ML\\New folder\\data_final.csv',sep='\t')
data.to_csv('E:\\Naresh IT\\project ML\\New folder\\data.csv',sep='\t')

#dependent values
y = data_final.RESPONSE.values

data_final.drop(columns=['RESPONSE'],inplace = True)

#independent variables
X = data_final.values

#splitting the data set in test and train part,train has 80% data and test has 20% data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#models building
#-----------------------------------------------------------------------------------
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
print(score) #75.5% accuracy

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3))

from sklearn import metrics
pd.crosstab(y_test,y_pred3)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred3)
roc_auc= metrics.auc(fpr,tpr)  #0.69

#-------------------------------------------------------------------------------------
#decision tree model
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier(max_depth = 4,random_state =42)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) #71.4

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

pd.crosstab(y_test,y_pred2)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred2)
roc_auc= metrics.auc(fpr,tpr) #0.64

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_model = AdaBoostClassifier(base_estimator =DecisionTreeClassifier(),learning_rate = 0.1, n_estimators = 10 )
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ada)
ada_model.score(X_test,y_test) #66% accuracy


from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier(learning_rate = 0.1,n_estimators = 100, subsample = 0.5,max_features = 0.15)
gbc_model.fit(X_train, y_train)

y_pred_gbc = gbc_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_gbc)

gbc_model.score(X_test,y_test) #76% accuracy
cm = confusion_matrix(y_test, y_pred_gbc)


pd.crosstab(y_test,y_pred_gbc)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred_gbc)
roc_auc= metrics.auc(fpr,tpr) #0.69
#---------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model1 = RandomForestClassifier(random_state = 0) 
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
model1.score(X_test,y_test) #76%

score = model1.score(X_test, y_test)
print(score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)


features=model1.feature_importances_

pd.crosstab(y_test,y_pred1)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred1)
roc_auc= metrics.auc(fpr,tpr) # auc= 0.7

#-----------------------------------------------------------------------------------------
#SVC model
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
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
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 100)
accuracies.mean() #76.7% accuracy
accuracies.std()
                
from sklearn.model_selection import GridSearchCV

parameters = [{'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
                           #n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr) #0.67

#-----------------------------------------------------------------
#chi-square test for checking which varibales are important
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

df  = pd.read_excel('Modeling.xlsx')
df.rename(columns={'RADIO/TV':'RADIO_TV'},inplace = True)
df['dummyCat'] = np.random.choice([0, 1], size=(len(df),), p=[0.5, 0.5])

#Initialize ChiSquare Class
cT = ChiSquare(df)

#Feature Selection
testColumns = ['CHK_ACCT','DURATION','EMPLOYMENT','HISTORY','MALE_SINGLE','OWN_RES','RADIO_TV','NEW_CAR','USED_CAR',
               'SAV_ACCT','REAL_ESTATE','PROP_UNKN_NONE','OTHER_INSTALL','RENT']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="RESPONSE" ) 

#---------------------------------------------------------------------------------------------------------------
#After doing chisquare test we got the important varibales 
#considering those important variables and building models
#In the duration there are lot of outliers so I have considered the maximum duration as 44    
#building the model by taking the duration values <=44

data = pd.read_excel('Modeling.xlsx')
data.drop(columns=['OBS#'],inplace = True)

data.describe()
core=data.corr()
data.DURATION.unique()

#considering only the values of durations <=44 because all the values beyond this is outliers and does 
#not produces good accuracy.
data = data[data.DURATION<=44]

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

#checking unique values in duration
data1.DURATION.unique()
np.sort(data1.DURATION.unique())

data1.info()
data1.CHK_ACCT=data1.CHK_ACCT.astype('str')
data1.HISTORY= data1.HISTORY.astype('str')
data1.SAV_ACCT= data1.SAV_ACCT.astype('str')
data1.EMPLOYMENT= data1.EMPLOYMENT.astype('str')


#separting the categorical and numerical data
data1_cat = data1.select_dtypes(exclude = [np.number])
data1_numerical= data1.select_dtypes(include = [np.number])

data1_cat_dummy=pd.get_dummies(data1_cat)
data1_cat_dummy.columns

#dropping the dummy variables whcih are not important based on the EDA
data1_cat_dummy.drop(columns=['CHK_ACCT_2'],inplace = True)
data1_cat_dummy.drop(columns=['HISTORY_0'],inplace = True)
data1_cat_dummy.drop(columns=['SAV_ACCT_3'],inplace = True)
data1_cat_dummy.drop(columns=['EMPLOYMENT_0'],inplace = True)

data1_final=pd.concat([data1_numerical,data1_cat_dummy],axis = 1)

#dependent values
y = data1_final.RESPONSE.values

data1_final.drop(columns=['RESPONSE'],inplace = True)

#independent variables
X = data1_final.values

#splitting the data set in test and train part,train has 80% data and test has 20% data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#------------------------------------------------------------------------------------
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

from sklearn import metrics
pd.crosstab(y_test,y_pred3)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred3)
roc_auc= metrics.auc(fpr,tpr)  #0.7

#---------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier(max_depth = 5,random_state =42)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) #69%

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

#------------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_model = AdaBoostClassifier(base_estimator =DecisionTreeClassifier(),learning_rate = 0.1, n_estimators = 10 )
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ada)
ada_model.score(X_test,y_test) #65.5% accuracy

from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier(learning_rate = 0.1,n_estimators = 100, subsample = 0.5,max_features = 0.15)
gbc_model.fit(X_train, y_train)

y_pred_gbc = gbc_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_gbc)

gbc_model.score(X_test,y_test) #78% accuracy
cm = confusion_matrix(y_test, y_pred_gbc)

pd.crosstab(y_test,y_pred_gbc)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred_gbc)
roc_auc= metrics.auc(fpr,tpr) #0.71

# ploting ROC curve
plt.title('ROC by using the gradientBoosting Classifier')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

#---------------------------------------------------------------
#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model1 = RandomForestClassifier(random_state = 0) 
model1.fit(X_train,y_train)
y_pred1 = model1.predict(X_test)
model1.score(X_test,y_test) #74%

score = model1.score(X_test, y_test)
print(score) #76% accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred1)

features=model1.feature_importances_

pd.crosstab(y_test,y_pred1)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred1)
roc_auc= metrics.auc(fpr,tpr) # auc= 0.7

# ploting ROC curve
plt.title('ROC by using the Random forest Classifier')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

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

from sklearn import metrics
pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr)

# ploting ROC curve
plt.title('ROC by using the Random forest Classifier')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

#with c=1000 and Linear kernel the accuracy is good which is 77% but this takes a lot of time to execute
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
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #76% accuracy  

from sklearn import metrics
pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr)  #0.677
    
plt.title('ROC by using the Random forest Classifier')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()
#--------------------------------------------------
#applying PCA and bilding the model
from sklearn.decomposition import PCA

X = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

pca = PCA(n_components=None)
fit= pca.fit(x)
fit.explained_variance_ratio_
fit.components_

pca = PCA(n_components=3)
X = pca.fit_transform(x) 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,)
classifier.fit(X_train, y_train)
y_pred3 = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)

score = classifier.score(X_test, y_test)
print(score) #73.3

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3))

from sklearn import metrics
pd.crosstab(y_test,y_pred3)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred3)
roc_auc= metrics.auc(fpr,tpr)  #0.5

plt.title('ROC by using the logistic by appling pca')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

#---------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model2 = DecisionTreeClassifier(max_depth = 5,random_state =42)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
model2.score(X_test,y_test) #69%

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

#------------------------------------------------------------------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_model = AdaBoostClassifier(base_estimator =DecisionTreeClassifier(),learning_rate = 0.1, n_estimators = 10 )
ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_ada)
ada_model.score(X_test,y_test) #65.5% accuracy

from sklearn.ensemble import GradientBoostingClassifier
gbc_model = GradientBoostingClassifier(learning_rate = 0.1,n_estimators = 100, subsample = 0.5,max_features = 0.15)
gbc_model.fit(X_train, y_train)

y_pred_gbc = gbc_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_gbc)

gbc_model.score(X_test,y_test) #78% accuracy
cm = confusion_matrix(y_test, y_pred_gbc)

pd.crosstab(y_test,y_pred_gbc)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred_gbc)
roc_auc= metrics.auc(fpr,tpr) #0.71

# ploting ROC curve
plt.title('ROC by using the gradientBoosting Classifier')
plt.plot(fpr,tpr, 'b' ,label = 'AUC =%0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

#----------------------------------------------------------------------
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
print("Accuracy: %.2f%%" % (accuracy * 100.0)) #68.5% 

from sklearn import metrics
pd.crosstab(y_test,y_pred)
fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
roc_auc= metrics.auc(fpr,tpr)  #0.51


