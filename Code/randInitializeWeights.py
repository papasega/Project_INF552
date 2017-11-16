from numpy import *

def randInitializeWeights(layers):

    num_of_layers = len(layers)
    epsilon = 0.12
    #epsilon=0.05

    Theta = []
    for i in range(num_of_layers-1):
        W = zeros((layers[i+1], layers[i] + 1),dtype = 'float64')
        # ====================== TODO ======================
        # Instructions: Initialize W randomly so that we break the symmetry while
        #               training the neural network.
        #
        W2 = random.uniform(low=-epsilon,high=epsilon,size=(layers[i+1], layers[i] + 1))
        W = W + W2
        Theta.append(W)

    return Theta
