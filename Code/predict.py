from numpy import *
from sigmoid import sigmoid

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels for each one of the instances

    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ TODO ================================
    # You need to return the following variables correctly
    p = zeros((1,m))
    # We don't need to optimize the predict function, so here we loop over training instances
    for i in range(m):
        a = X[i,:] # a stands for "activation"
        for l in range(num_layers-1):
            a = toRight(a,Theta[l],sigmoid) #to move from activation l to activation n+1
        assert(a.shape[0]==num_labels)
        p[0,i] = 1 if a > 0.5 else 0
    return p

def toRight(a,w,sigmoid):
    z = w[:,1:].dot(a)+w[:,0]
    o = sigmoid(z)
    return o
