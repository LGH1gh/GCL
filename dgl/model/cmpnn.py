import logging
import objgraph
import time
logger = logging.getLogger()
from data import get_bond_fdim, get_atom_fdim
import torch
import dgl
from torch import nn as nn
from dgl import nn as dglnn
from dgl import function as fn
from dgl.utils import expand_as_pair
from .utils import get_activation_function



def edge_message(edges: dgl.udf.EdgeBatch):
    """
    dgl.udf.EdgeBatch supports .src .dst .data .edges .batch_sie
    """
    # subtract reverse edge feature in (k-1) layer
    # from destination node feature in kth layer
    with torch.no_grad():
        rev_msg_indices = []
        for i in range(0, edges.batch_size(), 2):
            rev_msg_indices.extend([i+1, i])
        rev_msg_indices = torch.tensor(rev_msg_indices, device=edges.data['h_k'].device).long()
    rev_msg = torch.index_select(edges.data['h_k'], dim=0, index=rev_msg_indices)
    return {'m_k': edges.dst['h_k'] - rev_msg}


class MPNEncoderDGL(nn.Module):
    """
    Ref: Communicative Representation Learning on Attributed Molecular Graphs
    """
    def __init__(self,
            atom_fdim,
            bond_fdim,
            hidden_size,
            depth,
            bias,
            dropout=0.0,
            activation='relu'):
        super().__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.bias = bias
        self.dropout = dropout
        self.activation = activation
        # dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        # activation
        self.act_func = get_activation_function(activation)
        # all Linears
        self.w_i_atom = nn.Linear(atom_fdim, hidden_size, bias=bias)
        self.w_i_bond = nn.Linear(bond_fdim, hidden_size, bias=bias)
        self.w_h_deep = nn.ModuleDict({
            f'w_h_{d}': nn.Linear(hidden_size, hidden_size, bias=bias)
            for d in range(self.depth - 1)
            })
        self.w_o = nn.Linear(hidden_size, hidden_size)  # TODO size mismatch?
        self.linear = nn.Linear(hidden_size * 3, hidden_size, bias=bias)

        self.reducer = fn.sum
        self.reset_parameters()

    def forward(self, g, nodes_feature=None, edges_feature=None):
        """
        update_all = use msg to update node representation
        apply_nodes/edges = use node/edge features to compute new node/edge features 
                            with an user-defined function
        """
        with g.local_scope():
            # nodes (num_atoms x hidden_size)
            g.apply_nodes(
                lambda nodes: {'h_0': self.act_func(self.w_i_atom(nodes.data['attr']))}
                )  
            g.apply_nodes(
                lambda nodes: {'h_k': self.act_func(self.w_i_atom(nodes.data['attr']))}
                )  
            # edges (num_bonds x hidden_size)
            g.apply_edges(
                lambda edges: {'h_0': self.act_func(self.w_i_bond(edges.data['attr']))}
                ) 
            g.apply_edges(
                lambda edges: {'h_k': self.act_func(self.w_i_bond(edges.data['attr']))}
                )  
            for k in range(self.depth - 1):
                # 1. use edge msg to update node
                # collect h_k of incoming edges, reduce and update for dst node
                g.update_all(fn.copy_e('h_k', 'm'), fn.sum('m', 'h_sum'))
                g.update_all(fn.copy_e('h_k', 'm'), fn.max('m', 'h_max'))
                g.dstdata['m_k'] = g.dstdata['h_sum'] * g.dstdata['h_max']
                g.dstdata['h_k'] = g.dstdata['h_k'] + g.dstdata['m_k']

                # 2. use node msg to update edge
                g.apply_edges(edge_message)
                w_h_k = self.w_h_deep[f'w_h_{k}']
                g.apply_edges(lambda edges: {
                    'h_k': self.dropout_layer(self.act_func(edges.data['h_0'] + w_h_k(edges.data['m_k'])))
                    })
            # the K-th update
            g.update_all(fn.copy_e('h_k', 'm'), fn.sum('m', 'h_sum'))
            g.update_all(fn.copy_e('h_k', 'm'), fn.max('m', 'h_max'))
            g.dstdata['m_k'] = g.dstdata['h_sum'] * g.dstdata['h_max']

            g.dstdata['hidden'] = self.linear(torch.cat([
                g.dstdata['m_k'],  # agg_message
                g.dstdata['h_k'],  # message_atom
                g.dstdata['h_0'],  # input_atom
                ], dim=1))
            # skipping GRU
            g.dstdata['hidden'] = self.dropout_layer(self.act_func(self.w_o(g.dstdata['hidden'])))  # num_atoms x hidden
            # readout
            # z <- {h(v) for any v in V}
            z = dgl.readout_nodes(g, 'hidden', op='mean')  # B x H
            return z

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.activation)
        # weight
        nn.init.xavier_normal_(self.w_i_atom.weight, gain=gain)
        nn.init.xavier_normal_(self.w_i_bond.weight, gain=gain)
        for m in self.w_h_deep.values():
            nn.init.xavier_normal_(m.weight, gain=gain)
        nn.init.xavier_normal_(self.w_o.weight, gain=gain)
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        # bias
        if self.bias:
            nn.init.zeros_(self.w_i_atom.bias)
            nn.init.zeros_(self.w_i_bond.bias)
            for m in self.w_h_deep.values():
                nn.init.zeros_(m.bias)
            nn.init.zeros_(self.w_o.bias)
            nn.init.zeros_(self.linear.bias)


class CMPNN(nn.Module):
    """
    Ref: Communicative Representation Learning on Attributed Molecular Graphs
    """
    def __init__(self, args):
        super().__init__()
        self.atom_fdim = get_atom_fdim()
        self.bond_fdim = get_bond_fdim() + get_atom_fdim()
        self.hidden_size = args.hidden_dim
        self.depth = args.depth
        self.bias = args.bias
        self.dropout = args.dropout
        self.activation = args.activation
        self.sigmoid = nn.Sigmoid()
        # dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # activation
        self.act_func = get_activation_function(self.activation)
        
        self.encoder = MPNEncoderDGL(
                atom_fdim = self.atom_fdim,
                bond_fdim = self.bond_fdim,
                hidden_size = self.hidden_size,
                depth = self.depth,
                bias = self.bias,
                dropout = self.dropout,
                activation = self.activation
                )
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2, bias=self.bias) 
        self.fc2 = nn.Linear(self.hidden_size // 2, 1, bias=self.bias)

        self.reset_parameters()

    def forward(self, g, nodes_feature=None, edges_feature=None):
        z = self.encoder(g)  # B x H
        logits = self.fc2(self.act_func(self.fc1(z)))

        if not self.training:
            logits = self.sigmoid(logits)
        
        return logits

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.activation)
        # weight
        nn.init.xavier_normal_(self.fc1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight, gain=gain)
        # bias
        if self.bias:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)




        