import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F

from model.hyp_crossvit import *


class pyramid_trans_expr(nn.Module):
    def __init__(self):
        super(pyramid_trans_expr, self).__init__()
        self.pyramid_fuse = HyVisionTransformer(in_chans=49, q_chanel=49, embed_dim=512,
                                                depth=8, num_heads=8, mlp_ratio=2.,
                                                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)

        self.se_block = SE_block(input_dim=512)



    def forward(self, x_ir, x_face):
        y_hat = self.pyramid_fuse(x_ir, x_face)
        y_hat = self.se_block(y_hat)
        y_feat = y_hat

        return y_feat


class SE_block(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim, input_dim)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.relu(x1)
        x1 = self.linear2(x1)
        x1 = self.sigmod(x1)
        x = x * x1
        return x

