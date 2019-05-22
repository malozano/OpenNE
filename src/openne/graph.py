"""Graph utilities."""

# from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp

__author__ = "Zhang Zhengyan"
__email__ = "zhangzhengyan14@mails.tsinghua.edu.cn"


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0
        self.look_up_edge = {}  # (a,b) -> X
        self.look_back_edge = {}  # X -> (a,b)
        self.edge_size = 0
        self.directed = False
        self.directed_line = False

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def encode_edge(self):
        G = self.G
        look_up_edge = self.look_up_edge
        look_back_edge = self.look_back_edge

        if self.directed_line:
            for edge in G.edges():
                look_up_edge[edge] = self.edge_size
                look_back_edge[self.edge_size] = edge
                self.edge_size += 1
        else:
            G = G.to_undirected()
            for edge in G.edges():
                reversed_edge = (edge[1], edge[0])
                look_up_edge[edge] = self.edge_size
                look_up_edge[reversed_edge] = self.edge_size
                look_back_edge[self.edge_size] = edge
                self.edge_size += 1

    def to_line_labels(self, X, Y):
        look_back_edge = self.look_back_edge
        labels = dict(zip(X, Y))
        if self.directed_line:
            # Assign destination labels to the edges
            new_labels = [(str(n), labels[look_back_edge[n][1]]) for n in look_back_edge]
        else:
            # Mix labels from the origin and destination node for undirected edges
            new_labels = [(str(n), list(set(labels[look_back_edge[n][0]]).union(set(labels[look_back_edge[n][1]])))) for
                          n in look_back_edge]
        Xb, Yb = zip(*new_labels)

        return Xb, Yb

    def adjacency_matrix(self, nodelist=None):
        if nodelist is None:
            nodelist = self.G.nodes

        A = nx.adjacency_matrix(self.G, nodelist=nodelist)
        return A

    def line_adjacency_matrix(self, nodelist=None, weighted=False):
        look_back_edge = self.look_back_edge

        node_size = self.node_size
        look_up_dict = self.look_up_dict
        look_back_list = self.look_back_list
        G = self.G

        if nodelist is None:
            nodelist = look_back_edge.keys()

        B = sp.csr_matrix((len(nodelist), node_size), dtype=np.int8)
        for node in map(int, nodelist):
            B[node, look_up_dict[look_back_edge[node][0]]] = 1
            B[node, look_up_dict[look_back_edge[node][1]]] = 1

        if weighted:
            k = [(1.0 / (G.degree[node] - 1)) for node in look_back_list]
            K = sp.diags(k)
            C = B.dot(K.dot(B.transpose()))
        else:
            C = B.dot(B.transpose())

        C.setdiag(0)

        return C

    def line_adjacency_matrix_node_walk(self, nodelist=None):
        look_back_edge = self.look_back_edge

        node_size = self.node_size
        look_up_dict = self.look_up_dict
        look_back_list = self.look_back_list
        G = self.G

        if nodelist is None:
            nodelist = look_back_edge.keys()

        B = sp.csr_matrix((len(nodelist), node_size), dtype=np.int8)
        for node in map(int, nodelist):
            B[node, look_up_dict[look_back_edge[node][0]]] = 1
            B[node, look_up_dict[look_back_edge[node][1]]] = 1

        k = [(1.0 / (G.degree[node])) for node in look_back_list]
        K = sp.diags(k)
        A = nx.adjacency_matrix(G, nodelist=look_back_list)

        E = B.dot(K.dot(A.dot(K.dot(B.transpose()))))

        return E

    def edge_array(self):
        edge_array = np.zeros((len(self.look_up_edge.keys()), 2), dtype=np.int32)
        for i, edge in enumerate(self.look_up_edge.keys()):
            edge_array[i][0] = self.look_up_dict[edge[0]]
            edge_array[i][1] = self.look_up_dict[edge[1]]
        return edge_array

    def read_g(self, g):
        self.G = g
        self.encode_node()
        self.encode_edge()

    def read_adjlist(self, filename, directed_line=False):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self.G = nx.read_adjlist(filename, create_using=nx.DiGraph())
        for i, j in self.G.edges():
            self.G[i][j]['weight'] = 1.0

        self.directed_line = directed_line
        self.encode_node()
        self.encode_edge()

    def read_edgelist(self, filename, weighted=False, directed=False, directed_line=False):
        self.G = nx.DiGraph()
        self.directed = directed
        self.directed_line = True if directed else directed_line

        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = float(w)
                self.G[dst][src]['weight'] = float(w)

        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
        fin.close()

        self.node_size = 0
        self.edge_size = 0

        self.encode_node()
        self.encode_edge()

    def read_node_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['label'] = vec[1:]
        fin.close()

    def read_node_features(self, filename):
        fin = open(filename, 'r')
        for l in fin.readlines():
            vec = l.split()
            self.G.nodes[vec[0]]['feature'] = np.array(
                [float(x) for x in vec[1:]])
        fin.close()

    def read_node_status(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G.nodes[vec[0]]['status'] = vec[1]  # train test valid
        fin.close()

    def read_edge_label(self, filename):
        fin = open(filename, 'r')
        while 1:
            l = fin.readline()
            if l == '':
                break
            vec = l.split()
            self.G[vec[0]][vec[1]]['label'] = vec[2:]
        fin.close()
