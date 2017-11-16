from numpy import *

def roll_params(nn_params, layers):

    # Setup some useful variables
    num_layers = len(layers)
    Theta = []
    index = 0
    for i in range(num_layers -1):
        step = layers[i+1] * (layers[i]+1)
        Theta.append(reshape(nn_params[index:(index+step)],(layers[i+1], (layers[i] + 1))))
        
        index = index + step

    return Theta
