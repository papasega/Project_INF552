from numpy import *

def unroll_params(Theta):
    
    nn_params = reshape(Theta[0],(1,-1)).transpose()
    for i in range(1,len(Theta)):
        nn_params = concatenate((nn_params,reshape(Theta[i], (1, -1)).transpose()))
        
    nn_params = ndarray.flatten(nn_params)
    
    return nn_params
        
