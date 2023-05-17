import networkx as nx
import numpy as np

def spectral_clustering(G):
    A = nx.adjacency_matrix(G).toarray()
    A = A + A.T
    A[A > 1] = 1
    D = np.diag(np.sum(A, axis = 0))
    D1 = np.linalg.inv(D)
    L = D1@(D-A)
    w, v = np.linalg.eig(L)
    z_ = v[:, 1]
    z = z_ > 0
    return z