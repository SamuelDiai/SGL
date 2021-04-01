import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
import pandas as pd
from SGL import LearnGraphTopology
plots_dir = './plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

def load_data_cancer(df_cancer):
    X = np.array(df_cancer)
    SCM = np.dot((X - np.mean(X, axis = 0).reshape(1, -1)), (X - np.mean(X, axis = 0).reshape(1, -1)).T) / X.shape[0]
    return SCM

def Cancer(df_cancer, y_cancer, alpha, beta, k, n_iter):
    if not os.path.exists(os.path.join(plots_dir, 'cancer')):
        os.makedirs(os.path.join(plots_dir, 'cancer'))
    SCM = load_data_cancer(df_cancer)
    # estimate underlying graph
    sgl = LearnGraphTopology(SCM, n_iter=10, beta=beta, alpha=alpha)
    graph = sgl.learn_graph(k=k)

    # Build graph
    A = graph['adjacency']
    G = nx.from_numpy_matrix(A)
    pos = nx.spring_layout(G)
    fig = plt.figure(figsize=(12,12))
    # Color labels
    color_map = []
    color_dict = {'PRAD' : 'blue', 'LUAD' : 'red', 'BRCA' : 'green', 'KIRC' : 'orange', 'COAD' : 'purple'}
    for i in range(y_cancer.shape[0]):
        color_map.append(color_dict[y_cancer['Class'][i]])
    # Plot graph
    nx.draw(G, node_color=color_map, with_labels=True, pos = pos, font_weight='bold')
    plt.title("Learned graph for the cancer dataset k=%s n_iter=%s alpha=%.3f beta=%.3f" % (k , n_iter, alpha, beta))
    filename = os.path.join(plots_dir, 'cancer', 'graph')
    fig.savefig(filename)
    return graph
