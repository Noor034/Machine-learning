# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:22:21 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#importing data set

dataset = pd.read_csv('Social_Network_ads.csv')

X = dataset.iloc[:, [2,3]].values
#X =X.reshape(-1,1)
Y = dataset.iloc[:,4].values
#Y = Y.reshape (-1,1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.25 ,random_state = 0)


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)


from sklearn.svm import SVC

classifier = SVC(kernel= 'rbf' , random_state= 0)

classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)



