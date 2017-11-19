from read_dataset import read_dataset
from read_dataset import refine_dataset
from randInitializeWeights import randInitializeWeights
from unroll_params import unroll_params
from roll_params import roll_params
from costFunction import costFunction
from backwards import backwards
from predict import predict
import numpy as np
from scipy.optimize import *
from predict_proba import predict_proba
import csv

features, X, y, ids = read_dataset("train.csv")
refine_dataset(X)
# randomly select 60000 samples from original dataset
indices = np.random.randint(X.shape[0], size=200000)
X_training = X[indices,:]
y_training = y[indices]
#X_training = X_[:60000, :]
#y_training = y_[:60000]
X_test = X[np.where(y == 1)[0], :]
print("shape X_test : {}".format(X_test.shape[0]))
#X_test = X_[60000:, :]
#y_test = y_[60000:]

input_layer_side = len(features)
nb_labels = 1
hidden_layer_size_0 = 15
hidden_layer_size_1 = 40

# initialize neural network
layers = [input_layer_side, hidden_layer_size_0, nb_labels]

Theta = randInitializeWeights(layers)

# Unroll parameters
nn_weights = unroll_params(Theta)

# training neural network
print("\nTraining Neural Network... \n")

# try different values of the regularization factor
lambd = 3.0

res = fmin_l_bfgs_b(costFunction, nn_weights, fprime = backwards, args = (layers,  X_training, y_training, nb_labels, 1.0), maxfun = 50, factr = 1., disp = True)
Theta = roll_params(res[0], layers)

print("\nTesting Neural Network... \n")

pred  = predict(Theta, X_test)
print('\nAccuracy: ' + str(np.mean(pred == 1) * 100))


features, test_samples, id_list = read_dataset('test.csv', train=0)
nb_test_samples = test_samples.shape[0]

print("\nRunning Test Data from Kaggle ...\n")
test_result = predict_proba(Theta, test_samples)

assert(test_result.shape == id_list.shape)
result_file = open('kaggle.csv', 'w')
with result_file:
    writer = csv.writer(result_file)
    writer.writerow(['id', 'target'])
    for i in range(nb_test_samples):
        writer.writerow([int(id_list[i]), float(test_result[i])])

print("\nFinish writing file. \n")