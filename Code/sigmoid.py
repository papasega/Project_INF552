from numpy import *

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    #g = zeros(shape(z))

    # ============================= TODO ================================
    # Instructions: Compute sigmoid function evaluated at each value of z.
    g = 1/(1+exp(-z))
    #g = exp(z)/(1+exp(z))
    return g
