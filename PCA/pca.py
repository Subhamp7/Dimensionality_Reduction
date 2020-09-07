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
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#loading the dataset
data=load_breast_cancer()
dataset=pd.DataFrame(data['data'],columns=data['feature_names'])

#assigning the data into dependent and independent data
X=dataset.values
Y=data['target']

#scaling the dataset
sc=StandardScaler()
X=sc.fit_transform(X)

#splitting the data into test and training set
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.25, random_state=0)

#applying the pca for visualization
pca=PCA(n_components=dataset.shape[1])
X_train=pca.fit_transform(X_train)
ratio=(pca.explained_variance_ratio_)*100

#plotting the pca varaince and number of components
plt.grid(b=True, which='both', axis='both')
plt.xticks(np.arange(0, dataset.shape[1], step=1))
plt.plot(range(0,dataset.shape[1]),ratio)
plt.xlabel('no_of_components')
plt.ylabel('Ratio')
plt.show()

#plotting the pca componenets
plt.scatter(X[:,0],X[:,1],Y)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')

#applying PCA
pca=PCA(n_components=2)
X_train=pca.fit_transform(X_train)

#fitting LR
lr=LogisticRegression()
lr.fit(X_train, Y_train)

#predicting
X_test=pca.transform(X_test)
pred=lr.predict(X_test)

#checking the accuracy metrics
print("classification_report : \n", classification_report(pred, Y_test))
print("confusion_matrix : \n", confusion_matrix(pred, Y_test))
print("Accuracy : ", (accuracy_score(pred, Y_test))*100)