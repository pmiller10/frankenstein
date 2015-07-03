import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy

def polynomial(matrix, polynomial):
    out = copy.copy(matrix)
    for i in range(polynomial+1)[2:]:
        raised = np.power(matrix, i)
        out = np.concatenate((out, raised), axis=1)
    return out

def norm(matrix):
    min_max_scaler = MinMaxScaler()
    matrix = min_max_scaler.fit_transform(matrix)
    return matrix

def scale(matrix):
    scaler = StandardScaler()
    scaler.fit(matrix)
    matrix = scaler.transform(matrix)
    return matrix
