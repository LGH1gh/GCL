import torch
from torch import nn

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:

    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    # print(index_size)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    # print(suffix_dim)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    # print(final_size)
    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    target[index==0] = 0
    return target

def get_activation_function(activation: str) -> nn.Module:
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')