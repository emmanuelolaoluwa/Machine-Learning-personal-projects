import pandas as pd
import  numpy as np

Z0  = np.array([0.01,0.02,0.03,0.01,0.01,0,0])
Z1 = np.array([0.01,0.03,0.04,0.04,0.01,0,0])
Z2 = np.array([0.01,0.04,0.07,0.04,0.02,0.02,0.01])
Z3 = np.array([0,0.02,0.05,0.06,0.04,0.03,0.02])
Z4 = np.array([0,0,0.01,0.03,0.05,0.05,0.03])
Z5 = np.array([0,0,0,0.01,0.02,0.03,0.04])

M = np.array([Z0,Z1,Z2,Z3,Z4,Z5])
#expected 

col_mean = np.mean(M, axis = 0)
row_mean = np.mean(M, axis = 1)
print(col_mean)
print(row_mean)




















