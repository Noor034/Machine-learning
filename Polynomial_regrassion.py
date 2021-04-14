# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 11:33:46 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

#importing data set

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values
X =X.reshape(-1,1)
Y = dataset.iloc[:,2].values
Y = Y.reshape (-1,1)

#just linear regrassion

from sklearn.linear_model import LinearRegression

clf=LinearRegression()

clf.fit(X,Y)

#polynomial regrassion

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 4)

X_poly = poly.fit_transform(X)

clf2 = LinearRegression()

clf2.fit(X_poly,Y) 

#plot of linear regrassion

plt.scatter(X, Y, color = 'red')

plt.plot(X, clf.predict(X), color = 'blue')

plt.title('Truth vs Bluff(Linear regrassion)')

plt.xlabel('Position')
plt.ylabel('salary')

plt.show()

#plot of 

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X, Y, color = 'red')

plt.plot(X_grid, clf2.predict(poly.fit_transform(X_grid)), color = 'blue')

plt.title('Truth vs Bluff(polynomial regrassion)')

plt.xlabel('Position')
plt.ylabel('salary')

plt.show()

#predic with liner regrassion

clf.predict([[6.5]])

#predict with polynomial regrassion

clf2.predict(poly.fit_transform([[6.5]]))












