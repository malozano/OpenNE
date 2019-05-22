import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import adjusted_rand_score


class Clustering(object):

    def __init__(self, vectors, ground_truth=None, num_classes=5):
        self.nodes = sorted(vectors.keys())
        self.emb = [vectors[node] for node in self.nodes]
        self.labels = None
        self.labels_unique = None

        if ground_truth is not None:
            ground_truth = [sorted(ground_truth[node]) for node in self.nodes]
            self.ground_truth_labels_unique, self.ground_truth_labels = np.unique(ground_truth, return_inverse=True)
            self.num_classes = len(self.ground_truth_labels_unique)
            print('Setting number of clusters to {}'.format(self.num_classes))
        else:
            self.num_classes = num_classes
            self.ground_truth_labels = None
            self.ground_truth_labels_unique = None
            print('Setting number of clusters to default ({})'.format(self.num_classes))

    def fit(self):
        # bandwidth = estimate_bandwidth(self.emb, quantile=0.01)
        # ms = MeanShift(bandwidth=bandwidth)
        ms = AgglomerativeClustering(n_clusters=self.num_classes)
        ms.fit(self.emb)

        self.labels = ms.labels_
        self.labels_unique = np.unique(self.labels)

    def modularity(self, A, labels=None, use_ground_truth_labels=False):
        D_in = A.sum(axis=0)  # Input degree (sum rows)
        D_out = A.sum(axis=1)  # Output degree (sum columns)
        DD = np.matmul(D_out, D_in)
        vol = float(np.sum(D_in))
        Q = (A / vol) - (DD / (vol * vol))

        if labels is not None:
            labels_unique = np.unique(labels)
        elif use_ground_truth_labels:
            labels = self.ground_truth_labels
            labels_unique = self.ground_truth_labels_unique
        else:
            labels = self.labels
            labels_unique = self.labels_unique

        modularity = 0
        for label in labels_unique:
            indices = labels == label
            modularity += Q[np.ix_(indices, indices)].sum()

        return modularity

    def ari(self):
        return adjusted_rand_score(labels_true=self.ground_truth_labels, labels_pred=self.labels)

    def write_node_labels(self, filename):
        fout = open(filename, 'w')
        for x, y in zip(self.nodes, self.labels):
            fout.write("{} {}\n".format(x, y))
        fout.close()


def to_binary_matrix(labels):
    binarizer = MultiLabelBinarizer(sparse_output=True)
    binarizer.fit(labels)
    Y = binarizer.transform(labels)
    return Y
