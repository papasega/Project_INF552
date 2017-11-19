from numpy import *
from sigmoid import sigmoid

def predict_proba(Theta, X):
    m = X.shape[0]
    num_layers = len(Theta) + 1

    p = zeros(m)
    # We don't need to optimize the predict function, so here we loop over training instances
    for i in range(m):
        a = X[i, :]  # a stands for "activation"
        for l in range(num_layers - 1):
            a = toRight(a, Theta[l], sigmoid)  # to move from activation l to activation n+1
        assert (a.shape[0] == 1)
        p[i] = a
    return p


def toRight(a, w, sigmoid):
    z = w[:, 1:].dot(a) + w[:, 0]
    o = sigmoid(z)
    return o