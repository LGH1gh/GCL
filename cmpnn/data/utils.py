from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
import math
from torch.utils.data.dataloader import DataLoader
from .dataset import MoleculeDatapoint, MoleculeDataset, MPNNDataset
import logging
logger = logging.getLogger()

DATA_INFO = {
    'zinc15_250K_2D': {'columns': ['smiles']},
    'zinc15_1M_2D': {'columns': ['smiles']},
    'bbbp': {'task_num': 1, 'task_type': 'classification', 'columns': ['smiles', 'p_np']},
    'clintox': {'task_num': 2, 'task_type': 'classification', 'columns': ['smiles', 'FDA_APPROVED', 'CT_TOX']},
    'tox21': {'task_num': 12, 'task_type': 'classification', 'columns': ['smiles', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']}
}

def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0])

def load_data(data_dir, data_name, columns: List[str]):
    data_path = f'{data_dir}/{data_name}.csv'
    data = pd.read_csv(data_path)
    data = data[columns].values.tolist()

    data = MoleculeDataset([
        MoleculeDatapoint(
            line=line,
        ) for i, line in tqdm(enumerate(data), total=len(data))
    ])

    # Filter out invalid SMILES
    original_data_len = len(data)
    data = filter_invalid_smiles(data)

    if len(data) < original_data_len:
        logger.debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               split_size: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[MoleculeDataset,
                                               MoleculeDataset,
                                               MoleculeDataset]:
    assert len(split_size) == 3 and sum(split_size) == 1

    if split_type == 'random':
        data.shuffle(seed=seed)
        train_size = int(split_size[0] * len(data))
        train_val_size = int((split_size[0] + split_size[1]) * len(data))

        train = data[:train_size]
        val = data[train_size:train_val_size]
        test = data[train_val_size:]
    else:
        raise NotImplementedError(f'Not Supportted Split Type {split_type}')

    return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

def load_dataloader(args):
    data_info = DATA_INFO[args.data_name]

    data = load_data(args.data_dir, args.data_name, data_info['columns'])
    args.task_num = data_info['task_num']
    args.task_type = data_info['task_type']
    train_dataset, val_dataset, test_dataset = split_data(data=data, split_type=args.split_type, split_size=args.split_size, seed=args.seed)
    
    if args.feature_scaling:
        args.feature_scaler = train_dataset.normalize_features(replace_nan_token=0)
        val_dataset.normalize_features(args.feature_scaler)
        test_dataset.normalize_features(args.feature_scaler)
    else:
        args.feature_scaler = None

    train_dataset, val_dataset, test_dataset = MPNNDataset(train_dataset), MPNNDataset(val_dataset), MPNNDataset(test_dataset)
    
    args.train_steps_per_epoch = len(train_dataset) // args.batch_size
    args.test_steps_per_epoch = math.ceil(len(test_dataset) / args.batch_size)
    args.val_steps_per_epoch = math.ceil(len(val_dataset) / args.batch_size)
    logger.info(f'train data: {len(train_dataset)} | test data: {len(test_dataset)} | val data: {len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    return train_dataloader, test_dataloader, val_dataloader

def load_cl_dataloader(args):
    data_info = DATA_INFO[args.cl_data_name]

    data = load_data(args.data_dir, args.cl_data_name, data_info['columns'])
    dataset = MPNNDataset(data)
    args.cl_steps_per_epoch = len(dataset) // args.cl_batch_size
    return DataLoader(dataset, batch_size=args.cl_batch_size, num_workers=1, drop_last=True)
