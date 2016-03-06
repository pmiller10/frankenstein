import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing
import copy

def polynomial(matrix, polynomial):
    out = copy.copy(matrix)
    for i in range(polynomial+1)[2:]:
        raised = np.power(matrix, i)
        out = np.concatenate((out, raised), axis=1)
    return out
