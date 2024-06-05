import sys
import argparse
import yaml
import os
import math
import time
import pickle as pkl
import scipy as sp
from scipy import io
import numpy as np
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
import pandas as pd
import networkx as nx
from sklearn.preprocessing import label_binarize
import torch
from numpy.linalg import eig, eigh
from utils import seed_everything


def normalize_graph(g, power, norm_type):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** power)
    adj = np.dot(np.dot(deg, g), deg)
    if norm_type == 'laplacian':
        if power == -0.5:
            _deg = np.eye(g.shape[0])
        else:
            _deg = adj.sum(axis=1).reshape(-1)
            # _deg[_deg == 0.] = 1.0
            _deg = np.diag(_deg)
        res = _deg - adj
    elif norm_type == 'adjacency':
        res = adj
    else:
        raise NotImplementedError
    return res



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("node_raw_data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("node_raw_data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, labels


def generate_node_data(dataset):
    if dataset in ['cora', 'citeseer', 'pubmed']:

        adj, y = load_data(dataset)
        adj = adj.todense()

    elif dataset in ['chameleon', 'squirrel', 'actor', 'cornell', 'texas']:
        edge_df = pd.read_csv('node_raw_data/{}/'.format(dataset) + 'out1_graph_edges.txt', sep='\t')
        node_df = pd.read_csv('node_raw_data/{}/'.format(dataset) + 'out1_node_feature_label.txt', sep='\t')
        y = node_df[node_df.columns[2]]

        num_nodes = len(y)
        adj = np.zeros((num_nodes, num_nodes))

        source = list(edge_df[edge_df.columns[0]])
        target = list(edge_df[edge_df.columns[1]])

        for i in range(len(source)):
            adj[source[i], target[i]] = 1.
            adj[target[i], source[i]] = 1.

    else:
        raise NotImplementedError

    return adj, y


def homophily_test(adj, y):
    # adj = torch.ones_like(adj) * 0.001
    # adj = adj.abs()

    # # Global normalization
    # adj = adj / adj.sum()

    # # Degree normalizaion
    # deg = adj.sum(1)
    # deg[deg == 0.] = 1.0
    # deg = torch.diag(deg ** -0.5)
    # adj = deg @ adj @ deg

    y_re = (y + 1.0).repeat(y.shape[0], 1)
    _y_map = y_re - y_re.transpose(0, 1)
    mask_diff = torch.where(_y_map == 0.0, 0.0, 1.0)

    # # Reconstruction homophily 1
    # mask_same = torch.where(_y_map != 0.0, 0.0, 1.0)
    # y_smooth = (adj * mask_same).pow(2).sum() / (adj * mask_diff).pow(2).sum()

    # Reconstruction homophily 2
    y_smooth = (adj * mask_diff).pow(2).sum() / adj.pow(2).sum()

    return y_smooth


adj, y = generate_node_data('cora')
adj = normalize_graph(adj, -0.5, norm_type='adjacency')

adj = torch.FloatTensor(adj)
y = torch.LongTensor(y)
if len(y.size()) > 1:
    if y.size(1) > 1:
        y = torch.argmax(y, dim=1)
    else:
        y = y.view(-1)

print(homophily_test(adj, y))

