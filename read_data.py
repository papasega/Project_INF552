import numpy as np
from sklearn.model_selection import train_test_split
from pandas import read_csv
import math as mt

def read_data(filename, train=1):
    data = read_csv(filename)
    X = data.iloc[:,2:]
    y_ = data.iloc[:,1]

    m = X.shape[0]
    nb_features = X.shape[1]

    y = np.zeros((m, 2))
    y[np.arange(m), y_] = 1

    print("\nnumber of samples : {}\n".format(m))
    print("\nnumber of features : {}\n".format(nb_features))

    """
    PreProcess Data :
    
    -replace missing data with Gaussian
    -rescale data to [0,1]
    """
    print("\npreprocessing data ...\n")
    for i in range(nb_features):
        indices = np.where(X.values[:,i] != -1)[0]
        mean = np.mean(X.values[indices, i])
        var = np.var(X.values[indices, i])
        X.values[np.where(X.values[:,i] == -1)[0], i] = np.random.normal(mean, mt.sqrt(var), m - indices.shape[0])
        X.values[:,i] = (X.values[:,i] - np.min(X.values[:,i])) / (np.max(X.values[:,i]) - np.min(X.values[:,i]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print('\nDone download and preprocessing data!\n')

    return X_train, X_test, y_train, y_test

