# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 03:01:29 2017

@author: hp
"""


# Rescale data (between 0 and 1)
import pandas as pd 
import scipy as sc 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer


train_p2 =  pd.read_csv("C:/Users/hp/Desktop/Cours_3A/machnique learning 1/projet2/train.csv")

#colums = ['msno']

#train_p2 = train_p2.drop(colums, axis=1, inplace = True)

### separate array into input and output components
X = train_p2[:,0:2]
Y = train_p2[:,1]

rus = RandomUnderSampler()
X_res, Y_res = rus.fit_sample(X ,Y)
##
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
### summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

## Standardize data (0 mean, 1 stdev)
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])

## Normalize data (length of 1)
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
## summarize transformed data
np.set_printoptions(precision=3)
print(normalizedX[0:5,:])

## binarization
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
## summarize transformed data
np.set_printoptions(precision=3)
print(binaryX[0:5,:])