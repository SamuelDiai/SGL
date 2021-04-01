import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from .SGL import LearnGraphTopology
plots_dir = './plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    
## Experiments on basic datasets
def load_dataset_and_sgl(dataset):
    """ Plot two moons dataset and learn the graph using SGL
    n : number of nodes per cluster
    k : number of components"""
    # Create save path
    if not os.path.exists(os.path.join(plots_dir, dataset)):
        os.makedirs(os.path.join(plots_dir, dataset))
    # Create dataset

def load_dataset_and_sgl(dataset, k, k_sgl, n):
    """ Plot two moons dataset and learn the graph using SGL
        n : number of nodes per cluster
        k : number of components to generate in the datasets
        k_sgl : number of components to learn using SGL
    """
    # Create save path
    if not os.path.exists(os.path.join(plots_dir, dataset)):
        os.makedirs(os.path.join(plots_dir, dataset))
    # Create dataset
    if dataset == 'Two moons'
        assert(k == 2)
        X, y = make_moons(n_samples=n*k, noise=.05, shuffle=True)
    elif dataset == 'Blops':
        X, y = make_blobs(n_samples=n*k, centers=k, n_features=2, random_state=0)
    else :
        raise ValueError('%s is not a valid dataset ' % dataset)
    # dict to store position of nodes
    pos = {}
    for i in range(n*k):
        pos[i] = X[i]

    # compute sample correlation matrix
    S = np.dot(X, X.T)
    # estimate underlying graph
    sgl = LearnGraphTopology(S, n_iter=100, beta=0.1)
    graph = sgl.learn_graph(k=k)

    # build network
    A = graph['adjacency']
    G = nx.from_numpy_matrix(A)
    print('Graph statistics:')
    print('Nodes: ', G.number_of_nodes(), 'Edges: ', G.number_of_edges() )

    # normalize edge weights to plot edges strength
    all_weights = []
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight'])
    max_weight = max(all_weights)
    norm_weights = [3* w / max_weight for w in all_weights]
    norm_weights = norm_weights

    # plot graph
    fig = plt.figure(figsize=(15,15))
    nx.draw_networkx(G,pos, width=norm_weights)
    plt.title("Learned graph for %s dataset" % dataset)
    plt.suptitle('SGL components k_slg=%s, Real components k=%s' % (k_sgl, k))
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    filename = os.path.join(plots_dir, dataset, 'graph_%s_%s' % (k , k_slg, n))
    fig.savefig(filename)


def two_moons(n, k_sgl):
    """ Plot two moons dataset and learn the graph using SGL
    n : number of nodes per cluster
    k : number of components"""
    # Create save path
    load_dataset_and_sgl('Two moons', 2, k_slg, n)


def blops(n, k, k_sgl):
    """ Plot blops dataset and learn the graph using SGL
    n : number of nodes per cluster
    k : number of components"""
    # Create save path
    load_dataset_and_sgl('Blops', k, k_sgl, n)
