from typing import List
from torch import nn
import lmdb
import pickle
import dgl
import logging
import torch
from data import get_bond_fdim, get_atom_fdim
from .utils import get_activation_function
import dgl.function as fn
from dgl.nn import SetTransformerEncoder
from dgl.nn import SetTransformerDecoder


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = SetTransformerEncoder(get_atom_fdim(), 4, hidden_dim, hidden_dim)
        self.decoder = SetTransformerDecoder(get_atom_fdim(), 4, hidden_dim, hidden_dim, 1, 4)
        self.linear = nn.Linear(get_atom_fdim() * 4, hidden_dim)

    def forward(self, graph, nodes_feature, edges_feature):
        encode = self.encoder(graph, nodes_feature)
        decode = self.decoder(graph, encode)
        output = self.linear(decode)
        return output


class SetTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = Encoder(args.hidden_dim)

        projection = [
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        ]

        # Create projection model
        self.projection = nn.Sequential(*projection)

    def forward(self, graph, nodes_feature, edges_feature):
        return self.projection(self.encoder(graph, nodes_feature, edges_feature))
