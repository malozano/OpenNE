from time import time
import networkx as nx
import numpy as np
import math as mt
import scipy as sp
import numpy as np
from scipy import linalg
from scipy import sparse


class CTE(object):

    def __init__(self, graph, d):

        self._d = d
        self._method_name = "cte"
        self.g = graph
        self._node_num = graph.node_size
        self.learn_embedding()

    def learn_embedding(self):
        graph = self.g.G
        graph = graph.to_undirected()
        t1 = time()
        d = self._d

        print('Computing Commute Time Embedding')

        # Adjacency matrix
        A = nx.adjacency_matrix(graph)

        # Degree matrix power D^(-1/2) and volume
        n, m = A.shape

        diags = A.sum(axis=1)

        vol = diags.sum(axis=0)
        diags = np.float_power(diags, -1 / 2)

        D = sp.sparse.spdiags(diags.flatten(), [0], m, n)

        # Normalized Laplacian and eigen-decomposition
        L = nx.normalized_laplacian_matrix(graph)
        L = L.tocsc()

        num_eigen = min(n - 2, d)

        print('Computing eigenvalues')

        la, P = sparse.linalg.eigs(L, num_eigen, which='SM', sigma=0)
        la = np.float_power(la, -1 / 2)
        n, m = P.shape

        X = sp.sparse.spdiags(la.flatten(), [0], num_eigen, num_eigen)

        # P and X are the eigenvectors and eigenvalues matrices respectively
        Pt = P.transpose()
        srvol = mt.sqrt(vol)

        # Embedding E = srvol*X*Pt*D
        E = X * srvol
        E = E.dot(Pt)
        E = D.dot(E.T)

        t2 = time()

        print('Building vectors')

        vectors = {}
        look_back = self.g.look_back_list
        for i in range(E.shape[0]):
            vectors[look_back[i]] = E[i].real.tolist()
        self.vectors = vectors

        return vectors, (t2 - t1)

    def learn_embedding_dense(self):
        graph = self.g.G
        graph = graph.to_undirected()
        t1 = time()

        # Adjacency matrix
        print('Adjacency matrix')
        A = nx.adjacency_matrix(graph)
        A = A.todense()

        # Degree matrix power D^(-1/2) and volume
        print('Degree matrix')
        n, m = A.shape
        diags = A.sum(axis=1)
        vol = diags.sum(axis=0)
        diags = np.float_power(diags, -1 / 2)
        D = sp.sparse.spdiags(diags.flatten(), [0], m, n)
        D = D.todense()

        # Normalized Laplacian and eigen-decomposition
        print('Normalized Laplacian')
        L = nx.normalized_laplacian_matrix(graph)
        L = L.todense()
        la, P = linalg.eig(L)
        la = np.float_power(la, -1 / 2)
        n, m = P.shape
        X = sp.sparse.spdiags(la.flatten(), [0], m, n)
        X = X.todense()

        # P and X are the eigenvectors and eigenvalues matrices respectively
        print('Computing embedding')
        Pt = P.transpose()
        srvol = mt.sqrt(vol)

        # Embedding E = srvol*X*Pt*D
        E = X * srvol
        E = E.dot(Pt)
        E = E.dot(D)

        t2 = time()

        print('Building vectors')

        vectors = {}
        lim = self._d + 1
        look_back = self.g.look_back_list
        for i in range(E.shape[1]):
            vectors[look_back[i]] = E[1:lim, i].real.reshape(lim - 1).tolist()[0]
        self.vectors = vectors

        return vectors, (t2 - t1)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self._d))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node,
                                        ' '.join([str(x) for x in vec])))
        fout.close()
