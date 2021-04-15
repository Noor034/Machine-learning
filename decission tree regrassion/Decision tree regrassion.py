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


#fitting the regrassion model

from sklearn.tree import DecisionTreeRegressor
regrassor = DecisionTreeRegressor(random_state= 0)
regrassor.fit(X,Y)



#predict with new result

Y_pred = regrassor.predict([[6.5]])


#plot of decision regrassor

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1) 

plt.scatter(X, Y, color = 'red')

plt.plot(X_grid, regrassor.predict(X_grid), color = 'blue')

plt.title('Truth vs Bluff(decision regrassion)')

plt.xlabel('Position')
plt.ylabel('salary')

plt.show()