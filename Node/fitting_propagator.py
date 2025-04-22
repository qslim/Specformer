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


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, nheads, dropout):
        super(MultiheadAttention, self).__init__()
        self.mha_norm = nn.LayerNorm(embed_dim)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(embed_dim, nheads, dropout)

    def forward(self, x):
        mha_x = self.mha_norm(x)
        mha_x, attn = self.mha(mha_x, mha_x, mha_x)
        x = x + self.mha_dropout(mha_x)
        return x


class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=128, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, nonlinear='GELU', residual=True, is_f_tf=False, layer_nonlinear=True):
        super(Specformer, self).__init__()

        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, nclass)
        )

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 1)

        self.mha_filter = MultiheadAttention(hidden_dim, nheads, tran_dropout)
        # self.ffn_filter = FFN(hidden_dim, tran_dropout, nonlinear)

        self.residual = residual
        self.layer_nonlinear = layer_nonlinear

    def forward(self, e, u, x):
        ut = u.permute(1, 0)
        h = self.feat_encoder(x)

        x = h

        eig = self.eig_encoder(e)  # [N, d]
        eig = self.mha_filter(eig)
        # eig = self.ffn_filter(eig)
        new_e = self.decoder(eig)  # [N, m]

        utx = ut @ h
        h = new_e * utx
        h = u @ h

        for _ in range(self.nlayer - 1):
            utx = ut @ h
            h = new_e * utx
            h = u @ h

        if self.residual:
            h = h + x

        return h