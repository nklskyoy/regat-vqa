"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

This code is modified by Linjie Li from Jin-Hwa Kim's repository.
https://github.com/jnhwkim/ban-vqa
MIT License
"""
from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    """
    Simple class for non-linear fully connect network
    """
    def __init__(self, dims, act='ReLU', dropout=0, bias=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
            layers.append(weight_norm(linear_layer, dim=None))
            if act is not None and act != '':
                activation = getattr(nn, act)()
                layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        linear_layer = nn.Linear(dims[-2], dims[-1], bias=bias)
        layers.append(weight_norm(linear_layer, dim=None))
        if act is not None and act != '':
            activation = getattr(nn, act)()
            layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits 

