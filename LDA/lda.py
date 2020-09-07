# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:52:54 2020

@author: subham
"""
#loading the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

#applying the pca
lda= LDA(n_components=1)#since Y has only two projections(True and False)
X_train =lda.fit_transform(X_train, Y_train)
ratio=(lda.explained_variance_ratio_)*100

#fitting LR 
lr=LogisticRegression()
lr.fit(X_train, Y_train)

#predicting
X_test=lda.transform(X_test)
pred=lr.predict(X_test)

#checking the accuracy metrics
print("classification_report : \n", classification_report(pred, Y_test))
print("confusion_matrix : \n", confusion_matrix(pred, Y_test))
print("Accuracy : ", (accuracy_score(pred, Y_test))*100)