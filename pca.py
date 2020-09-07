# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 15:48:01 2020

@author: subham
"""

#loading the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#loading the dataset
data=load_breast_cancer()
dataset=pd.DataFrame(data['data'],columns=data['feature_names'])

#scaling the dataset
sc=StandardScaler()
dataset=sc.fit_transform(dataset)

#applying the pca
pca=PCA(n_components=dataset.shape[1])
X=pca.fit_transform(dataset)
ratio=(pca.explained_variance_ratio_)*100

#plotting the pca varaince and number of components
plt.grid(b=True, which='both', axis='both')
plt.xticks(np.arange(0, dataset.shape[1], step=1))
plt.plot(range(0,dataset.shape[1]),ratio)
plt.xlabel('no_of_components')
plt.ylabel('Ratio')
plt.show()

#plotting the pca componenets
plt.scatter(X[:,0],X[:,1],c=data['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')