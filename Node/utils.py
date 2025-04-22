import time
import math
import random
import numpy as np
import scipy as sp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset


# def y_smoothness(e, u, y):
#     # print(u.shape)
#     y = y.float()
#     # print(y.shape)
#     utx = u.permute(1, 0) @ y.unsqueeze(-1)
#     # print(utx.shape)
#     _aggregated = e.unsqueeze(-1) * utx
#     aggregated = u @ _aggregated
#     # print(aggregated.shape)
#
#     lap_smooth = (y * aggregated.squeeze()).sum()
#
#     return lap_smooth


def mask_diff_computation(y):
    y_re = (y + 1.0).repeat(y.shape[0], 1)
    _y_map = y_re - y_re.transpose(0, 1)
    mask_diff = torch.where(_y_map == 0.0, 0.0, 1.0)

    # _y = (y + 1.0).unsqueeze(-1)
    # _y_map = _y.pow(-1) @ _y.transpose(0, 1)
    # mask_diff = torch.where(_y_map == 1.0, 0.0, 1.0)

    return mask_diff


def reconstruction_smoothness(mask_diff, filter=None, u=None, adj=None):
    if adj is not None:
        pass
        # adj = torch.ones_like(adj) * 0.001
        # adj = adj.abs()

        # # Global normalization
        # adj = adj / adj.sum()
    elif filter is not None and u is not None:
        # filter = torch.rand_like(filter) / filter.shape[0]
        adj = u @ (filter.unsqueeze(-1) * u.permute(1, 0))
        # adj = torch.ones_like(adj) * 0.001
        # adj = adj.abs()

        # Global normalization
        # adj = adj / adj.sum()

    adj = adj * (torch.ones_like(adj) - torch.eye(adj.shape[0], device=adj.device))

    # # Degree normalizaion
    # deg = adj.sum(1)
    # deg[deg == 0.] = 1.0
    # deg = torch.diag(deg ** -0.5)
    # adj = deg @ adj @ deg

    # # Reconstruction homophily 1
    # mask_same = torch.where(_y_map != 0.0, 0.0, 1.0)
    # y_smooth = (adj * mask_same).pow(2).sum() / (adj * mask_diff).pow(2).sum()

    # Reconstruction homophily 2
    # adj_2 = adj.pow(2)
    # y_smooth = (adj_2 * mask_diff).sum() / adj_2.sum()
    # y_smooth = (adj * mask_diff).abs().sum() / adj.abs().sum()
    y_smooth = ((adj * mask_diff).abs().sum() / mask_diff.sum()) / (adj.abs().sum() / (adj.shape[0] ** 2))
    # y_smooth = ((adj_2 * mask_diff).sum(1) / adj_2.sum(1)).mean()

    # # Reconstruction homophily 3
    # adj_2 = (adj @ adj).pow(2)
    # y_smooth = (adj_2 * mask_diff).sum() / adj_2.sum()

    return y_smooth.item()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(dataset, y, nclass, seed=0):
    
    if dataset in ['arxiv', 'products']:
        dataset = DglNodePropPredDataset('ogbn-' + dataset)
        split = dataset.get_idx_split()
        train, valid, test = split['train'], split['valid'], split['test']
        return train, valid, test

    elif dataset == 'penn':
        split = np.load('node_raw_data/fb100-Penn94-splits.npy', allow_pickle=True)[0]
        train, valid, test = split['train'], split['valid'], split['test']
        return train, valid, test

    else:
        y = y.cpu()

        percls_trn = int(round(0.6 * len(y) / nclass))
        val_lb = int(round(0.2 * len(y)))

        indices = []
        for i in range(nclass):
            index = (y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0), device=index.device)]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_index = rest_index[:val_lb]
        test_index = rest_index[val_lb:]

        return train_index, valid_index, test_index

