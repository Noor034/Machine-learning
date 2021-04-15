# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:41:59 2021

@author: User
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

#elbow method for number of clusters

from sklearn.cluster import KMeans 

wcss = []

for i in range(1,11):
    
     Kmeans =KMeans(n_clusters = i ,init = 'k-means++' ,max_iter = 300 , n_init = 10 , random_state = 0)
     Kmeans.fit(X)
     wcss.append(Kmeans.inertia_)
     
     
plt.plot(range(1,11),wcss) 

plt.title('The elbow method')

plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()   
     

#applying rght kmeans to the mall 

Kmeans =KMeans(n_clusters = 5 ,init = 'k-means++' ,max_iter = 300 , n_init = 10 , random_state = 0)

Y_kmeans =Kmeans.fit_predict(X)   
     
     
#visualising the clusters

plt.scatter(X[Y_kmeans == 0, 0] ,X[Y_kmeans == 0,1 ] , s = 100 ,c='red', label = 'Carefull')
plt.scatter(X[Y_kmeans == 1, 0] ,X[Y_kmeans == 1,1 ] , s = 100 ,c='blue', label = 'standerds')
plt.scatter(X[Y_kmeans == 2, 0] ,X[Y_kmeans == 2,1 ] , s = 100 ,c='green', label = 'target')       
plt.scatter(X[Y_kmeans == 3, 0] ,X[Y_kmeans == 3,1 ] , s = 100 ,c='cyan', label = 'careless') 
plt.scatter(X[Y_kmeans == 4, 0] ,X[Y_kmeans == 4,1 ] , s = 100 ,c='black', label = 'sensible') 

plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


