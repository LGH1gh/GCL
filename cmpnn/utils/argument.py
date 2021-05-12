from argparse import ArgumentParser, Namespace
from ast import parse
from typing import Tuple, List
import torch
def parse_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--data_name', type=str, default='tox21')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--feature_scaling', type=bool, default=True)
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--split_size', type=Tuple, default=(0.8, 0.1, 0.1))
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--hidden_dim', type=str, default=300)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='ReLU')

    parser.add_argument('--pn_generator', type=str, default='Dropout')
    parser.add_argument('--contrastive_loss', type=str, default='NCESoftmax')

    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--ffn_hidden_dim', type=int, default=300)
    parser.add_argument('--lr_num', type=int, default=1)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--final_lr', type=float, default=1e-4)
    parser.add_argument('--init_lr', type=float, default=1e-4)

    parser.add_argument('--cl_data_name', type=str, default='zinc15_250K_2D')
    parser.add_argument('--cl_max_lr', type=float, default=1e-5)
    parser.add_argument('--cl_final_lr', type=float, default=1e-5)
    parser.add_argument('--cl_init_lr', type=float, default=1e-5)
    parser.add_argument('--cl_batch_size', type=int, default=1250)
    parser.add_argument('--metric', type=str, default='auc')

    args = parser.parse_args()
    return args


