import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    # We transpose each matrix in Theta for easier calculation, because
    # we will treat X directly as a whole matrix and we won't loop over each training instance.
    # It's faster to tranpose Theta than to transpose X.
    for l in range(num_layers-1):
        Theta[l] = np.matrix.transpose(Theta[l])

    # You need to return the following variables correctly
    #J = 0;

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = np.zeros((m,num_labels))
    #for k in range(m): (SLOW)
        #yv[k,y[k]]=1
    yv[(np.arange(m),y)]=1


    # ================================ TODO ================================
    # In this point calculate the cost of the neural network (feedforward)
    # We won't loop over each training instance, we treat X as a whole matrix
    X2 = X + 0 # for safety reason, we would like to duplicate X to X2.
    for l in range(num_layers-1):
        z = X2.dot(Theta[l][1:,:]) + Theta[l][0,:]
        X2 = sigmoid(z)
    J = jo(yv,X2) #jo is the cost function, view below

    R = 0
    for l in range(num_layers-1): #regularization
        essen = Theta[l][1:,:]
        R = R + np.sum(essen*essen)

    return J/m + lambd*R/(2*m)
"""
def toRight(a,w,sigmoid): #this function is no longer necessary
    z = w[:,1:].dot(a)+w[:,0]
    o = sigmoid(z)
    return o
"""
def jo(y,r):
    return np.sum(-y*np.log(r)-(1-y)*np.log(1-r))
