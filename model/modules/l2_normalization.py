import torch
from torch import nn as nn


class L2Normalization(nn.Module):
    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, inp):
        if inp.shape[0] > 1:
            inp = inp.squeeze()
        return inp.div(torch.norm(inp, dim=1).view(-1, 1))

    def __repr__(self):
        return self.__class__.__name__
