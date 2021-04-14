# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:06:05 2021

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

#feature scalling beacuse svr does not have this default
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

#fitting the svr regrassion model

from sklearn.svm import SVR

regrassor = SVR(kernel = 'rbf')
regrassor.fit(X,Y)



#predict with new

Y_pred = sc_Y.inverse_transform(regrassor.predict(sc_X.transform(np.array([[6.5]]))))


#plot of svr

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1) 

plt.scatter(X, Y, color = 'red')

plt.plot(X_grid, regrassor.predict(X_grid), color = 'blue')

plt.title('Truth vs Bluff(SV regrassion)')

plt.xlabel('Position')
plt.ylabel('salary')

plt.show()