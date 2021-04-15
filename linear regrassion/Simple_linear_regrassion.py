# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:59:34 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#importing data set

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3 ,random_state = 0)


from sklearn.linear_model import LinearRegression

clf=LinearRegression()

clf.fit(X_train,Y_train)


#predict the test set results

Y_pred = clf.predict(X_test)

#visualising the training set

plt.scatter(X_train, Y_train, color = 'red')

plt.plot(X_train, clf.predict(X_train), color = 'blue')

plt.title('Salary vs Experience(Training set)')

plt.xlabel('Years of exp')
plt.ylabel('salary')

plt.show()


#visualising the test set

plt.scatter(X_test, Y_test, color = 'red')

plt.plot(X_train, clf.predict(X_train), color = 'blue')

plt.title('Salary vs Experience(test set)')

plt.xlabel('Years of exp')
plt.ylabel('salary')

plt.show()










