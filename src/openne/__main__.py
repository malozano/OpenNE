from __future__ import print_function
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from .graph import *
from . import node2vec
from . import edge2vec
from . import netmf
from . import PRUNE
from .classify import Classifier, read_node_label, write_node_label
from .clustering import Clustering
from . import line
from . import tadw
from .gcn import gcnAPI
from . import lle
from . import cte
from . import hope
from . import lap
from . import gf
from . import sdne
from .grarep import GraRep
import time
import ast


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output',
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--directed-line', action='store_true',
                        help='Treat line graph as directed (only for edge2vec and edgeWalk).')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', required=True, choices=[
        'node2vec',
        'edge2vec',
        'deepWalk',
        'edgeWalk',
        'netmf',
        'prune',
        'line',
        'gcn',
        'grarep',
        'tadw',
        'lle',
        'cte',
        'hope',
        'lap',
        'gf',
        'sdne'
    ], help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--output-label-file', default='',
                        help='The output file of generated edge labels (only for edge2vec and edgeWalk)')
    parser.add_argument('--output-cluster-file', default='',
                        help='Uses agglomerative clustering on node embeddings and generates new class labels')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--multi-clf-ratio', action='store_true',
                        help='Train with ratios [10%, 20%, 30$, 40%, 50%, 60%, 70%, 80%, 90%]')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification (if not multi-ratio)')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training LINE')
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='Dropout rate (1 - keep probability)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix')
    parser.add_argument('--hidden', default=16, type=int,
                        help='Number of units in hidden layer 1')
    parser.add_argument('--kstep', default=4, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--lamb', default=0.2, type=float,
                        help='lambda is a hyperparameter in TADW')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--alpha', default=1e-6, type=float,
                        help='alhpa is a hyperparameter in SDNE')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in SDNE')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in SDNE')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in SDNE')
    parser.add_argument('--bs', default=200, type=int,
                        help='batch size of SDNE')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer, the last number is the '
                             'dimension of the output node representation')
    args = parser.parse_args()

    if args.method != 'gcn' and not args.output:
        print("No output filename. Exit.")
        exit(1)

    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")

    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input, directed_line=args.directed_line)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed, directed_line=args.directed_line)

    if args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, p=args.p, q=args.q, window=args.window_size)
    elif args.method == 'edge2vec':
        model = edge2vec.Edge2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size)
    elif args.method == 'netmf':
        model = netmf.NetMF(graph=g, window=args.window_size, dim=args.representation_size)
    elif args.method == 'prune':
        model = PRUNE.PRUNE(graph=g, dim=args.representation_size, batchsize=128)
    elif args.method == 'line':
        if args.label_file and not args.no_auto_save:
            model = line.LINE(g, epoch=args.epochs, rep_size=args.representation_size, order=args.order,
                              label_file=args.label_file, clf_ratio=args.clf_ratio)
        else:
            model = line.LINE(g, epoch=args.epochs,
                              rep_size=args.representation_size, order=args.order)
    elif args.method == 'deepWalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'edgeWalk':
        model = edge2vec.Edge2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
    elif args.method == 'tadw':
        # assert args.label_file != ''
        assert args.feature_file != ''
        g.read_node_label(args.label_file)
        g.read_node_features(args.feature_file)
        model = tadw.TADW(
            graph=g, dim=args.representation_size, lamb=args.lamb)
    elif args.method == 'gcn':
        assert args.label_file != ''
        assert args.feature_file != ''
        g.read_node_label(args.label_file)
        g.read_node_features(args.feature_file)
        model = gcnAPI.GCN(graph=g, dropout=args.dropout,
                           weight_decay=args.weight_decay, hidden1=args.hidden,
                           epochs=args.epochs, clf_ratio=args.clf_ratio)
    elif args.method == 'grarep':
        model = GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)
    elif args.method == 'lle':
        model = lle.LLE(graph=g, d=args.representation_size)
    elif args.method == 'cte':
        model = cte.CTE(graph=g, d=args.representation_size)
    elif args.method == 'hope':
        model = hope.HOPE(graph=g, d=args.representation_size)
    elif args.method == 'sdne':
        encoder_layer_list = ast.literal_eval(args.encoder_list)
        model = sdne.SDNE(g, encoder_layer_list=encoder_layer_list,
                          alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
                          batch_size=args.bs, epoch=args.epochs, learning_rate=args.lr)
    elif args.method == 'lap':
        model = lap.LaplacianEigenmaps(g, rep_size=args.representation_size)
    elif args.method == 'gf':
        model = gf.GraphFactorization(g, rep_size=args.representation_size,
                                      epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)
    t2 = time.time()
    print('Total embedding time', t2-t1)

    if args.method != 'gcn':
        print("Saving embeddings...")
        model.save_embeddings(args.output)

    x, y = None, None
    if args.label_file and args.method != 'gcn':
        vectors = model.vectors
        x, y = read_node_label(args.label_file)
        if args.method == 'edge2vec' or args.method=='edgeWalk':
            x, y = model.graph.to_line_labels(x, y)
        if args.output_label_file:
            write_node_label(x, y, args.output_label_file)

        if args.multi_clf_ratio:
            for ratio in np.arange(0.1, 1.0, 0.1):
                print("Training classifier using {:.2f}% nodes...".format(ratio * 100))
                clf = Classifier(vectors=vectors, clf=LogisticRegression())
                clf.split_train_evaluate(x, y, ratio)
        else:
            print("Training classifier using {:.2f}% nodes...".format(args.clf_ratio * 100))
            clf = Classifier(vectors=vectors, clf=LogisticRegression())
            clf.split_train_evaluate(x, y, args.clf_ratio)

    if args.output_cluster_file and args.method != 'gcn':
        print("Clustering embeddings ...")
        vectors = model.vectors

        if x is None:
            clustering = Clustering(vectors=vectors)
        else:
            clustering = Clustering(vectors=vectors, ground_truth=dict(zip(x, y)))
        clustering.fit()

        if args.method == 'edge2vec' or args.method == 'edgeWalk' or args.linegraph:
            A = g.line_adjacency_matrix_node_walk(nodelist=clustering.nodes)
        else:
            A = g.adjacency_matrix(nodelist=clustering.nodes)

        clustering_results = dict()
        clustering_results['modularity'] = clustering.modularity(A)

        if x is not None:
            clustering_results['ground_truth_modularity'] = clustering.modularity(A, use_ground_truth_labels=True)
            clustering_results['ari'] = clustering.ari()

        print(clustering_results)

        clustering.write_node_labels(args.output_cluster_file)


if __name__ == "__main__":
    random.seed(32)

    np.random.seed(32)
    main(parse_args())
