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
import objgraph
import time
logger = logging.getLogger()

class CMPNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.node_feature_dim = get_atom_fdim()
        self.edge_feature_dim = get_bond_fdim()
        self.hidden_dim = args.hidden_dim
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.args = args

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.activation_function = get_activation_function(args.activation)
        self.W_i_atom = nn.Linear(self.node_feature_dim, self.hidden_dim, bias=self.bias)
        self.W_i_bond = nn.Linear(self.edge_feature_dim, self.hidden_dim, bias=self.bias)

        self.W_h_atom = nn.Linear(self.hidden_dim + self.edge_feature_dim, self.hidden_dim, bias=self.bias)

        for depth in range(self.depth-1):
            self._modules[f'W_h_{depth}'] = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)

        self.W_o = nn.Linear((self.hidden_dim)*2, self.hidden_dim)

        # self.gru = BatchGRU(self.hidden_dim)
        self.lr = nn.Linear(self.hidden_dim*3, self.hidden_dim, bias=self.bias)

    def forward(self, graph):
        graph.apply_nodes(lambda nodes: {'node_feature_origin': self.activation_function(self.W_i_atom(nodes.data['attr']))})
        graph.apply_nodes(lambda nodes: {'node_feature' : nodes.data['node_feature_origin']})

        graph.apply_edges(lambda edges: {'edge_feature_origin': self.activation_function(self.W_i_bond(edges.data['attr']))})
        graph.apply_edges(lambda edges: {'edge_feature' : edges.data['edge_feature_origin']})

        edges = graph.edges()
        for depth in range(self.depth - 1):
            graph.send_and_recv(edges, fn.copy_e('edge_feature', 'edge_feature'), fn.sum('dmpnn_hidden', 'dpmnn_aggregate'))

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.env = lmdb.open(f'{args.data_dir}/{args.cl_data_name}', map_size=int(1e12), max_dbs=1, readonly=True)
        self.graphs_db = self.env.open_db('graph'.encode())
        logger.info(self.env.stat())

    def forward(self, batch_idx: List[int]):
        graphs = []
        with self.env.begin() as txn:
            for idx in batch_idx:
                graphs.append(pickle.loads(txn.get(str(idx.item()).encode(), db=self.graphs_db)))
        graph_batch = dgl.batch(graphs)
        del graph_batch
        del graphs
        return 0

class FFN4Test(nn.Module):
    def __init__(self, args, data_info):
        super().__init__()
        first_linear_dim = args.hidden_dim
        activation = get_activation_function(args.activation)
        self.task_type = data_info['task_type']

        if self.task_type == 'classification':
            self.sigmoid = nn.Sigmoid()
        elif self.task_type == 'multiclass':
            self.softmax = nn.Softmax(dim=2)
        else:
            raise ValueError(f'The task type ({self.task_type}) are not supported.')

        ffn = [
            nn.Linear(first_linear_dim, args.ffn_hidden_dim),
            activation,
            nn.Linear(args.ffn_hidden_dim, data_info['task_num'])
        ]

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def forward(self, hidden):

        output = self.ffn(hidden)

        if self.task_type == 'classification' and not self.training:
            output = self.sigmoid(output)
        if self.task_type == 'multiclass':
            output = output.reshape((output.size(0), self.task_num, -1)) # batch size x num targets x num classes per target
            if not self.training:
                output = self.softmax(output) # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss

        return output

