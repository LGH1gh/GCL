from argparse import Namespace
import random
from typing import Callable, List, Union
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from rdkit import Chem

from .scaler import StandardScaler
from rdkit import Chem, RDLogger
import logging
logger = logging.getLogger()
RDLogger.DisableLog('rdApp.*') 


class MoleculeDatapoint:
    def __init__(self,
                 line: List[str],
                 args: Namespace = None):

        self.args = args
        self.smiles = line[0]  # str
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.targets = [float(x) if x != '' else None for x in line[1:]]
        self.features = None

    def set_features(self, features: np.ndarray):
        self.features = features

    def num_tasks(self) -> int:
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        self.targets = targets


class MoleculeDataset(Dataset):
    def __init__(self, data: List[MoleculeDatapoint]):
        self.data = data
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

    def compound_names(self) -> List[str]:
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:
        return [d.smiles for d in self.data]
    
    def mols(self) -> List[Chem.Mol]:
        return [d.mol for d in self.data]

    def features(self) -> List[np.ndarray]:
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:
        return [d.targets for d in self.data]

    def num_tasks(self) -> int:
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler
    
    def set_targets(self, targets: List[List[float]]):
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        self.data.sort(key=key)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return self.data[item]

class MPNNDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self) -> int:
        return len(self.dataset.data)
    
    def __getitem__(self, index):
        return self.dataset.data[index].smiles, torch.FloatTensor(self.dataset.data[index].targets)

