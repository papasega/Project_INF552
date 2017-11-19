import numpy as np

def read_dataset(filename, train = 1):
    data = np.genfromtxt(filename, delimiter=',')
    print("nb of training samples : {}".format(data.shape[0] - 1))

    if train:
        features = data[0, 2:]
        X = data[1 :, 2 :]
        y = data[1 :, 1].astype(int)
        ids = data[1 :, 0]
        return features, X, y, ids
    features = data[0, 1:]
    X = data[1 :, 1 :]
    ids = data[1 :, 0]
    return features, X, ids

def refine_dataset(X):
    m = X.shape[0]
    nb_features = X.shape[1]
    for i in range(nb_features):
        avg = X[:, i].dot(X[:, i] != -1) / (X[:, i] != -1).sum()
        X[np.where(X[:, i] == 1), i] = avg
        X[:, i] = X[:, i] / X[:, i].max()
    print("done refining dataset")
