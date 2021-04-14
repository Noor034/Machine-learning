# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:09:10 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#importing data set

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values



from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])


ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

#avoiding the dummy varriable trap

X=X[:, 1:]


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2 ,random_state = 0)


from sklearn.linear_model import LinearRegression

clf=LinearRegression()

clf.fit(X_train,Y_train)


Y_pred = clf.predict(X_test)

#building optimal model using backward elimination

import statsmodels.api as sm

X = np.append(arr = np.ones((50,1)).astype(int) ,values = X , axis = 1)

#backward regrassion

X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)

clf_OLS = sm.OLS(endog =Y ,exog =X_opt).fit()

clf_OLS.summary()


X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)

clf_OLS = sm.OLS(endog =Y ,exog =X_opt).fit()

clf_OLS.summary()


X_opt = np.array(X[:, [0,3, 4, 5]], dtype=float)

clf_OLS = sm.OLS(endog =Y ,exog =X_opt).fit()

clf_OLS.summary()


X_opt = np.array(X[:, [0,3,5]], dtype=float)

clf_OLS = sm.OLS(endog =Y ,exog =X_opt).fit()

clf_OLS.summary()

X_opt = np.array(X[:, [0,3]], dtype=float)

clf_OLS = sm.OLS(endog =Y ,exog =X_opt).fit()

clf_OLS.summary()









