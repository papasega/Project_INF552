import numpy as np
from cv import cv
import time
cv_values = np.loadtxt("cv_values.txt").astype(int)
for i in range(cv_values.shape[0]):
    start_t = time.time()
    g = cv(cv_values[i,0],cv_values[i,1],cv_values[i,2])
    end_t = time.time()
    h = np.array([cv_values[i,0],cv_values[i,1],cv_values[i,2],g,end_t-start_t])
    np.savetxt("line"+str(i)+".txt",h)
