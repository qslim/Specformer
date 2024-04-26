import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim, nheads, dropout):
        super(Transformer, self).__init__()
        self.mha_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.mha_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(embed_dim, nheads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, embed_dim, embed_dim)

    def forward(self, x):
        mha_x = self.mha_norm(x)
        mha_x, attn = self.mha(mha_x, mha_x, mha_x)
        x = x + self.mha_dropout(mha_x)

        ffn_x = self.ffn_norm(x)
        ffn_x = self.ffn(ffn_x)
        x = x + self.ffn_dropout(ffn_x)

        return x


class SpecLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none',
                 nheads=1, tran_dropout=0.0):
        super(SpecLayer, self).__init__()
        self.transformer = Transformer(ncombines, nheads, tran_dropout)

    def forward(self, x):
        x = self.transformer(x)
        return x


class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(Specformer, self).__init__()

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

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

        self.transformer = Transformer(hidden_dim, nheads, tran_dropout)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none':
            self.layers = nn.ModuleList(
                [SpecLayer(nheads + 1, nclass, prop_dropout, norm=norm, nheads=nheads, tran_dropout=tran_dropout) for i
                 in range(nlayer)])
            self.norm = None
        else:
            self.layers = nn.ModuleList(
                [SpecLayer(nheads + 1, hidden_dim, prop_dropout, norm=norm, nheads=nheads, tran_dropout=tran_dropout)
                 for i in range(nlayer)])
            if norm == 'layer':  # Arxiv
                self.norm = nn.LayerNorm(hidden_dim)
            elif norm == 'batch':  # Penn
                self.norm = nn.BatchNorm1d(hidden_dim)
            else:  # Others
                self.norm = None
    
    def forward(self, e, u, x):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm is None:
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig = self.eig_encoder(e)  # [N, d]

        eig = self.transformer(eig)

        new_e = self.decoder(eig)  # [N, m]

        x = h

        for conv in self.layers:
            utx = ut @ h
            h = conv(new_e * utx)

            h = u @ h

            if self.norm is None:
                h = F.gelu(h)
            else:
                h = self.norm(h)
                h = F.relu(h)

        h = h + x

        if self.norm is None:
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h

