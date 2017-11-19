from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)

    # You need to return the following variables correctly
    Theta_grad = [zeros(w.shape) for w in Theta]
    # We transpose each matrix in Theta for easier calculation, because
    # we will treat X directly as a whole matrix and we won't loop over each training instance.
    # It's faster to transpose theta than to transpose X.
    for l in range(num_layers-1):
        Theta[l] = matrix.transpose(Theta[l])

    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((m, num_labels))
    if num_labels == 1:
        yv[where(y == 1), 0] = 1
    else:
        yv[(arange(m), y)] = 1

    # ================================ TODO ================================
    # In this point implement the backpropagaition algorithm
    # We won't loop over each training instance, we treat X as a whole matrix

    A = [] # list of all activations
    Z = []; Z.append([]) # list of all z
    X2 = X+0 # for safety reason, we would like to duplicate X to X2.
    A.append(X2)
    for l in range(num_layers-1):
        z = X2.dot(Theta[l][1:,:]) + Theta[l][0,:]
        Z.append(z)
        X2 = sigmoid(z)
        A.append(X2)

    for l in range(num_layers-1):
        Theta[l] = matrix.transpose(Theta[l])

    delta=A[num_layers-1]-yv
    for l in reversed(range(num_layers-1)): #BACKpropagation
        # Here we use Einstein summation to get an array of outer products of delta[l+1] and A[l] over all training instances
        Theta_grad[l][:,1:] = sum(einsum('ij,ik->ijk',delta,A[l]),axis=0)
        Theta_grad[l][:,0] = sum(delta,axis=0)
        if (l==0):
            break
        delta = (delta.dot( Theta[l][:,1:] )) * sigmoidGradient(Z[l])

    for l in range(num_layers-1):
        Theta_grad[l] /= m
        Theta_grad[l][:,1:] += lambd/m*Theta[l][:,1:] #regularization
    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad
