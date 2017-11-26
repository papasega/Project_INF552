import numpy as np
from cv import cv
import time
cv_values = np.loadtxt("cv_values.txt").astype(int)
# In the cv_values file, we put values of (C,v,k) used for testing

for i in range(cv_values.shape[0]):
    start_t = time.time()
    g = cv(cv_values[i,0],cv_values[i,1],cv_values[i,2])
    end_t = time.time()
    h = np.array([cv_values[i,0],cv_values[i,1],cv_values[i,2],g,end_t-start_t])
    # In h there are all information related to the ith test and its result.
    np.savetxt("line"+str(i)+".txt",h)
    # the result of the ith test will be saved to the file "linei.txt"
