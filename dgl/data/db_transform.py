import pandas as pd
import os
import pickle
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from pandas.core.indexing import check_bool_indexer
import torch
from rdkit import Chem, RDLogger
import logging
logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*') 
import lmdb

from .featurization import smiles2dgl


DATA_INFO = {
    'zinc15_250K_2D': {'columns': ['smiles']},
    'zinc15_1M_2D': {'columns': ['smiles']},
    'zinc15_10M_2D': {'columns': ['smiles']},
}

if __name__ == '__main__':
    data_dir = '../../../data'
    data_name = 'zinc15_10M_2D'
    data_info = DATA_INFO[data_name]
    data_path = f'{data_dir}/{data_name}.csv'

    chunk_size = 1
    data = pd.read_csv(data_path, chunksize=chunk_size, iterator=True)
    env = lmdb.open(f'{data_dir}/{data_name}', map_size=int(1e12), max_dbs=1)
    for idx, chunk in enumerate(data):
        print(idx)
        graphs = list(chunk[data_info['columns'][0]].apply(smiles2dgl))
        graphs_db = env.open_db('graph'.encode())
        txn = env.begin(write=True)
        txn.put(str(idx).encode(), pickle.dumps(graphs[0]), db=graphs_db)
        txn.commit()
    env.close()
