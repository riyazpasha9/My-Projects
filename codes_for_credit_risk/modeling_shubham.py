%matplotlib
import pandas as pd
import numpy as np
import os
import seaborn as sns

os.chdir('E:programming material\\data science\\naveen_sir_material\\AI\\Datasets\\riyaz')

data = pd.read_csv('ModelingData.txt',sep='\t')

data.info()
data.isna().sum() #last two rows are having nan

data.dropna(inplace=True)
data.drop(columns=['OBS#'],inplace=True)
                 
summary = data.describe()

# given problem statement CHK_ACCT, history, sav_acct, employment,present resident, job are categorical data
# so we have to convert them from numerical to catagorical variables.

data[['CHK_ACCT','HISTORY','SAV_ACCT','EMPLOYMENT','PRESENT_RESIDENT','JOB']] = data[['CHK_ACCT','HISTORY','SAV_ACCT','EMPLOYMENT','PRESENT_RESIDENT','JOB']].astype('category')

# creating dummy variables
data = pd.get_dummies(data)
data.drop(columns=['CHK_ACCT_3.0','HISTORY_4.0','SAV_ACCT_4.0','EMPLOYMENT_4.0','PRESENT_RESIDENT_4.0','JOB_3.0'],inplace=True)


# Exploratory Data Analysis (EDA) #######################

sns.countplot(x='RESPONSE',data=data)
sns.countplot(x='EDUCATION',data=data)

sns.boxplot(data.AMOUNT)
sns.distplot(data.AMOUNT, hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


sns.boxplot(y=data.DURATION)
sns.distplot(data.DURATION, hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


sns.boxplot(y=data.AGE)
sns.distplot(data.AGE, hist=True, kde=True, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


corr = data.corr()
sns.pairplot(data[['DURATION','AMOUNT','AGE']], palette='tab20',size=6)


# EDA ends here #############################################

# models function
def models(x,y):
    from sklearn.linear_model import LogisticRegression
    from sklearn import model_selection

    kfold = model_selection.KFold(n_splits=10)
    model = LogisticRegression()
    result1= model_selection.cross_val_score(model,x,y,cv=kfold)
    #print('log reg accuracy : ',result1.mean())
    

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import model_selection

    kfold = model_selection.KFold(n_splits=10)
    model = DecisionTreeClassifier(max_depth=2)
    result2= model_selection.cross_val_score(model,x,y,cv=kfold)
    #print('d tree accuracy : ',result2.mean())

    from sklearn.ensemble import RandomForestClassifier
    from sklearn import model_selection
    
    kfold = model_selection.KFold(n_splits=10)
    model = RandomForestClassifier(n_estimators=100,max_features=3)
    result3= model_selection.cross_val_score(model,x,y,cv=kfold)
    #print('random forest accuracy : ',result3.mean()) 

    from sklearn.svm import SVC
    from sklearn import model_selection

    kfold = model_selection.KFold(n_splits=10)
    model = SVC()
    result4= model_selection.cross_val_score(model,x,y,cv=kfold)
    #print('svm accuracy',result4.mean()) 
    
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn import model_selection

    kfold = model_selection.KFold(n_splits=10)
    model5 = GradientBoostingClassifier(n_estimators=100)
    result5= model_selection.cross_val_score(model5,x,y,cv=kfold)
    #print('sg boost accuracy : ',result.mean()) 

    from xgboost import XGBClassifier
    from sklearn import model_selection
    
    kfold = model_selection.KFold(n_splits=10)
    model6 = XGBClassifier(n_estimators=100)
    result6= model_selection.cross_val_score(model6,x,y,cv=kfold)
   # print('xgboost accuracy : ',result.mean()) 
    
    print('\nlog reg accuracy : ',result1.mean())  
    print('decision tree accuracy : ',result2.mean())
    print('random forest accuracy : ',result3.mean())
    print('svm accuracy : ',result4.mean())
    print('sg boost accuracy : ',result5.mean())
    print('xgboost accuracy : ',result6.mean())
    
   
# passing all variables#######################
x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

models(x,y)
'''
log reg accuracy :  0.751
decision tree accuracy :  0.7089999999999999
random forest accuracy :  0.745
svm accuracy :  0.6950000000000001
sg boost accuracy :  0.7390000000000001
xgboost accuracy :  0.757
'''

#################################  Feature reduction ########

# multicollinearity check using vif ####################################
x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

def vif_func(x):
    VIF = pd.DataFrame()
    VIF["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    VIF["features"] = x.columns
    return VIF

vif = vif_func(x)
x.drop(columns=['OWN_RES'],inplace=True)
vif = vif_func(x)
x.drop(columns=['AGE'],inplace=True)
vif = vif_func(x)
x.drop(columns=['NUM_DEPENDENTS'],inplace=True)
vif = vif_func(x)
x.drop(columns=['INSTALL_RATE'],inplace=True)
vif = vif_func(x)
x.drop(columns=['NUM_CREDITS'],inplace=True)
vif = vif_func(x)


# model building

models(x,y)

'''
log reg accuracy :  0.7619999999999999
decision tree accuracy :  0.7089999999999999
random forest accuracy :  0.743
svm accuracy :  0.687
sg boost accuracy :  0.758
xgboost accuracy :  0.7630000000000001
'''
#################################

# select k best
x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

from sklearn.feature_selection import SelectKBest,chi2,f_regression
model = SelectKBest(score_func=chi2,k=30) 
fit = model.fit(x,y)
fit.scores_

x_kbest = fit.transform(x)

models(x_kbest,y)

'''
log reg accuracy :  0.753
decision tree accuracy :  0.7089999999999999
random forest accuracy :  0.7529999999999999
svm accuracy :  0.6990000000000001
sg boost accuracy :  0.754
xgboost accuracy :  0.759
'''

# RFE 
x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logreg = LogisticRegression()

rfe = RFE(logreg,35) 
fit = rfe.fit(x,y)

x_rfe = fit.transform(x)

models(x_rfe,y)

'''
log reg accuracy :  0.75
decision tree accuracy :  0.6900000000000001
random forest accuracy :  0.735
svm accuracy :  0.7299999999999999
sg boost accuracy :  0.758
xgboost accuracy :  0.7530000000000001

'''

# lasso reg feature selection
'''
x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

from sklearn.linear_model import Lasso
from sklearn import model_selection

model = Lasso(alpha=0.1)
model.fit(x,y)
pred = model.predict(x)

coeff = pd.DataFrame()
coeff['features']=x.columns 
coeff['lasso_coeff'] = model.coef_

'''

x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

from glmnet_python import glmnet; from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

model = glmnet(x = x.copy().values, y = y.copy().values, family = 'binomial')
model.fit(x)
model.glmnetcoef

x = x[['CHK_ACCT_0.0','CHK_ACCT_1.0','CHK_ACCT_2.0','DURATION','AMOUNT','SAV_ACCT_0.0','SAV_ACCT_1.0','SAV_ACCT_2.0','SAV_ACCT_3.0','INSTALL_RATE','OTHER_INSTALL']]


models(x,y)

'''
log reg accuracy :  0.7389999999999999
decision tree accuracy :  0.7089999999999999
random forest accuracy :  0.711
svm accuracy :  0.691
sg boost accuracy :  0.7460000000000001
xgboost accuracy :  0.753

'''

from sklearn.decomposition import PCA

x = data.drop(columns=['RESPONSE'])
y= data.RESPONSE

pca = PCA(n_components=None)
fit= pca.fit(x)
fit.explained_variance_ratio_
fit.components_

pca = PCA(n_components=3)
new_x = pca.fit_transform(x) 

models(new_x,y)

'''
log reg accuracy :  0.705
decision tree accuracy :  0.703
random forest accuracy :  0.666
svm accuracy :  0.699
sg boost accuracy :  0.6950000000000001
xgboost accuracy :  0.697
'''