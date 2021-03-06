import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from SGL import LearnGraphTopology
plots_dir = './plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def animals(k, n_iter, alpha, beta):
    if not os.path.exists(os.path.join(plots_dir, 'animals')):
        os.makedirs(os.path.join(plots_dir, 'animals'))
    X_animals = np.load('data/animals_data.npy')
    animals_names = np.load('data/animals_name.npy')
    animals_features = np.load('data/animals_features.npy')
    #from sgll.sgl import LearnGraphTopolgy
    X = np.array(X_animals)

    SCM = np.dot(X, X.T) / X.shape[0]
    SCM = 1/3 * np.eye(SCM.shape[0]) + SCM

    # estimate underlying graph
    sgl = LearnGraphTopology(SCM, n_iter=n_iter, alpha=alpha, beta = beta)
    graph = sgl.learn_graph(k=k)

    A = graph['adjacency']
    G = nx.from_numpy_matrix(A)

    # normalize edge weights to plot edges strength
    all_weights = []
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight'])
    max_weight = max(all_weights)
    norm_weights = [3* w / max_weight for w in all_weights]
    norm_weights = norm_weights

    mapping = {}
    for i in range(animals_names.shape[0]):
        mapping[i] = animals_names[i, 0]

    G = nx.relabel_nodes(G, mapping)
    fig = plt.figure(figsize=(12,12))
    nx.draw(G, with_labels=True, font_weight='bold', width=norm_weights)
    plt.title("Learned graph for the animal dataset k=%s n_iter=%s alpha=%.3f beta=%.3f" % (k , n_iter, alpha, beta))
    filename = os.path.join(plots_dir, 'animals', 'graph')
    fig.savefig(filename)
    return graph
