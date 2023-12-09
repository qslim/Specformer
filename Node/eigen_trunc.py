import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none':
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':   # Arxiv
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':  # Penn
            self.norm = nn.BatchNorm1d(ncombines)
        else:                  # Others
            self.norm = None

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x


class Specformer(nn.Module):

    def __init__(self, num_eigen, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(Specformer, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # for arxiv & penn
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none':
            self.layers = nn.ModuleList([SpecLayer(2, nclass, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            self.layers = nn.ModuleList([SpecLayer(2, hidden_dim, prop_dropout, norm=norm) for i in range(nlayer)])

        # self.filter = nn.Parameter(torch.empty((num_eigen, 1)))
        # nn.init.normal_(self.filter, mean=1.0, std=0.0)
        # nn.init.kaiming_normal_(self.filter)
        # nn.init.kaiming_uniform_(self.filter, a=math.sqrt(5))
        # nn.init.xavier_uniform_(self.filter)
        # nn.init.xavier_normal_(self.filter)
        self.filter = nn.Parameter(torch.ones((num_eigen, 1)))
        print('num_eigen:', num_eigen)

    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        filter = self.filter
        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h
            basic_feats.append(u @ (filter * utx))  # [N, d]
            basic_feats = torch.stack(basic_feats, axis=1)
            h = conv(basic_feats)

        if self.norm == 'none':
            return h, filter
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h, filter

