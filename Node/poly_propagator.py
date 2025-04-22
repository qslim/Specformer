import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SineEncoding(nn.Module):
    def __init__(self, degree):
        super(SineEncoding, self).__init__()
        self.eig_w = nn.Linear(degree, 1)
        self.degree = degree

    def forward(self, e):
        poly_basis = torch.vander(e, N=self.degree, increasing=True)
        return self.eig_w(poly_basis)


class Specformer(nn.Module):

    def __init__(self, nclass, nfeat, nlayer=1, hidden_dim=7, nheads=1,
                 tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, nonlinear='GELU', residual=True, is_f_tf=False, layer_nonlinear=True):
        super(Specformer, self).__init__()
        self.nlayer = nlayer

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, nclass)
        )

        self.eig_encoder = SineEncoding(hidden_dim)

        self.residual = residual
        self.layer_nonlinear = layer_nonlinear

    def forward(self, e, u, x):
        ut = u.permute(1, 0)
        h = self.feat_encoder(x)

        x = h

        new_e = self.eig_encoder(e)  # [N, d]

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