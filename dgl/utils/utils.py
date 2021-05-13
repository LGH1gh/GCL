import random
import numpy as np
import math
import torch
from argparse import Namespace
from torch.optim import Adam, Optimizer
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Union, Callable
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, init_lr) -> Optimizer:
    params = [{'params': model.parameters(), 'lr': init_lr, 'weight_decay': 0}]
    return Adam(params)

def build_lr_scheduler(optimizer: Optimizer, steps_per_epoch, init_lr, max_lr, final_lr, args: Namespace, total_epochs: List[int] = None) -> _LRScheduler:
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epoch_num] * args.lr_num,
        steps_per_epoch=steps_per_epoch,
        init_lr=[init_lr],
        max_lr=[max_lr],
        final_lr=[final_lr]
    )

def prc_auc(targets: List[int], preds: List[float]) -> float:
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets: List[float], preds: List[float]) -> float:
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse
    
    if metric =='mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score
    
    if metric == 'accuracy':
        return accuracy
    
    if metric == 'cross_entropy':
        return log_loss

    raise ValueError(f'Metric "{metric}" not supported.')

def get_loss_func(data_info) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if data_info['task_type'] == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    if data_info['task_type'] == 'regression':
        return nn.MSELoss()
    
    if data_info['task_type'] == 'multiclass':
        return nn.CrossEntropyLoss()
    
    raise ValueError(f'Dataset type "{data_info["task_type"]}" not supported.')

