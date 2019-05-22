#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05
# TODO:

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import networkx as nx
import argparse
import logging
import theano
from theano import tensor as T

logger = logging.getLogger(__name__)
theano.config.exception_verbosity = 'high'


class NetMF(object):

    def __init__(self, graph, window=10, dim=128, rank=256, negative=1.0, large=False):

        self.size = dim
        self.graph = graph

        # Scipy sparse adjacency matrix
        nodes = list(graph.G.nodes)
        A = nx.adjacency_matrix(graph.G, nodelist=nodes)

        if large:
            emb = netmf_large(A, window, dim, rank, negative)
        else:
            emb = netmf_small(A, window, dim, rank, negative)

        print('Embedding', emb.shape)

        self.vectors = {}
        for n in range(len(nodes)):
            self.vectors[str(nodes[n])] = emb[n]
        # self.vectors[word] = word2vec.wv[word]

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()


def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x * (1 - x ** window) / (1 - x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f", np.max(evals), np.min(evals))
    return evals


def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    # evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU


def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol / b)
    f = theano.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(theano.config.floatX))
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
                np.count_nonzero(Y))
    return sparse.csr_matrix(Y)


def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def netmf_large(A, window=10, dim=128, rank=256, negative=1.0):
    logger.info("Running NetMF for a large window size...")
    logger.info("Window size is set to be %d", window)
    # obtain graph volume
    vol = float(A.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=rank, which="LA")

    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,
                                                  window=window,
                                                  vol=vol, b=negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    # logger.info("Save embedding to %s", args.output)
    # np.save(args.output, deepwalk_embedding, allow_pickle=False)
    return deepwalk_embedding


def direct_compute_deepwalk_matrix(A, window, b):
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i + 1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = theano.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(theano.config.floatX))
    return sparse.csr_matrix(Y)


def netmf_small(A, window=10, dim=128, rank=256, negative=1.0):
    logger.info("Running NetMF for a small window size...")
    logger.info("Window size is set to be %d", window)
    # directly compute deepwalk matrix
    deepwalk_matrix = direct_compute_deepwalk_matrix(A, window=window, b=negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=dim)
    # logger.info("Save embedding to %s", args.output)
    # np.save(args.output, deepwalk_embedding, allow_pickle=False)
    return deepwalk_embedding
