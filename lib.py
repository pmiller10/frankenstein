import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
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


def k_means(matrix, n):
    if not n > 0:
        return matrix
    model = KMeans(n_clusters=n)
    model.fit(matrix)
    new_matrix = []
    for m in matrix:
        p = model.predict(m)
        cluster = p[0]
        clusters = [0. for i in range(n)]
        clusters[cluster] = 1.
        m = list(m)  # TODO this will break numpy multi dimensional arrays
        m = m + clusters
        new_matrix.append(m)
    return new_matrix
