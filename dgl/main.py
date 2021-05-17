from dgl.utils import data
from utils import parse_argument, set_seed, build_lr_scheduler, build_optimizer, get_metric_func, get_loss_func
from data import load_dataloader, load_cl_dataloader
from model import SetTransformer, CMPNN, FFN4Test, ContrastiveLoss
from typing import List
import numpy as np
from tqdm import tqdm
import lmdb
import pickle
import torch
import logging
import dgl
import time
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()

def train_encoder(epoch_idx, encoder, criterion, encoder_optimizer, encoder_scheduler, data_info, dataloader, args):
    encoder.train()
    losses = []
    with tqdm(enumerate(dataloader), desc=f'Epoch {epoch_idx}', total=data_info['train_steps_per_epoch'], ncols=100) as batch:
        for idx, x in batch:
            encoder_optimizer.zero_grad()
            graphs = []
            with args.env.begin() as txn:
                for idx in x:
                    graphs.append(pickle.loads(txn.get(str(idx.item()).encode(), db=args.graphs_db))) 
            graph_batch = dgl.batch(graphs).to(args.device)

            nodes_feature = graph_batch.ndata['attr']
            edges_feature = graph_batch.edata['attr']
            y_hat = encoder(graph_batch, nodes_feature, edges_feature)

            loss = criterion(y_hat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
            batch.set_postfix(loss=loss.item())
            encoder_optimizer.step()
            encoder_scheduler.step()
            losses.append(loss.item())
    return np.mean(losses)

def train_classifer(encoder, classifier, dataloader, classifier_optimizer, classifier_schedule, criterion, args):
    encoder.eval()
    classifier.train()
    for idx, (x, y) in enumerate(dataloader):
        mask = torch.Tensor([[not val.isnan() for val in col] for col in y]).to(args.device)
        x = x.to(args.device)
        y = torch.Tensor([[0 if val.isnan() else val for val in col] for col in y]).to(args.device)
        classifier_optimizer.zero_grad()
        with torch.no_grad():
            x_hat = encoder(x, x.ndata['attr'], x.edata['attr'])
        y_hat = classifier(x_hat)
        loss = criterion(y_hat, y)
        loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
        loss = loss.sum() / mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.1)
        classifier_optimizer.step()
        classifier_schedule.step()


def evaluate_concat(encoder, classifier, data_info, dataloader, metric_func, args):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        target, pred = [], []
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(args.device)
            y_hat = classifier(encoder(x, x.ndata['attr'], x.edata['attr'])).reshape(y.size())
            pred.append(y_hat.cpu())
            target.append(y)
        pred = torch.cat(pred)
        target = torch.cat(target)
        valid_preds = [[] for _ in range(data_info['task_num'])]
        valid_targets = [[] for _ in range(data_info['task_num'])]
        for i in range(data_info['task_num']):
            for j in range(len(pred)):
                if not target[j][i].isnan():  # Skip those without targets
                    valid_preds[i].append(pred[j][i])
                    valid_targets[i].append(target[j][i])
        
        results = []
        for i in range(data_info['task_num']):
            if data_info['task_type'] == 'classification':
                mark = False
                if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                    mark = True
                    logger.debug(f'Warning: Found a task {data_info["data_name"]} with targets all 0s or all 1s')
                if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                    mark = True
                    logger.debug('Warning: Found a task with predictions all 0s or all 1s')
                if mark:
                    results.append(float('nan'))
                    continue
                results.append(metric_func(valid_targets[i], valid_preds[i]))
        return np.nanmean(results)


def evaluate_encoder(epoch_idx, encoder, data_info, train_dataloader, val_dataloader, metric_func, max_patience, args):
    current_patience, best_result = 0, 0 
    classifier = FFN4Test(args, data_info).to(args.device)
    classifier_loss = get_loss_func(data_info).to(args.device)
    classifier_optimizer = build_optimizer(classifier, args.init_lr)
    classifier_scheduler = build_lr_scheduler(classifier_optimizer, data_info, args.init_lr, args.max_lr, args.final_lr, args)
    while True:
        train_classifer(encoder, classifier, train_dataloader, classifier_optimizer, classifier_scheduler, classifier_loss, args)
        result = evaluate_concat(encoder, classifier, data_info, val_dataloader, metric_func, args)
        if best_result < result:
            best_result = result
            current_patience = 0
        else:
            current_patience += 1
        if current_patience == max_patience:
            break
    logger.info(f'Epoch {epoch_idx}\'s {data_info["data_name"]} {args.metric} score: {best_result}')


def main():
    logging.basicConfig(format='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO, filename='result.log')
    logger.info('#'*100)
    args = parse_argument()
    set_seed(args.seed)
    args.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader_list, val_dataloader_list, test_dataloader_list, data_info_list = [], [], [], []
    for name in args.data_name:
        train_dataloader, val_dataloader, test_dataloader, data_info = load_dataloader(args, name)
        train_dataloader_list.append(train_dataloader)
        val_dataloader_list.append(val_dataloader)
        test_dataloader_list.append(test_dataloader)
        data_info_list.append(data_info)
    data_num = len(args.data_name)

    cl_dataloader, cl_data_info = load_cl_dataloader(args)
    args.env = lmdb.open(f'{args.data_dir}/{args.cl_data_name}', map_size=int(1e12), max_dbs=1, readonly=True)
    args.graphs_db = args.env.open_db('graph'.encode())

    encoder = CMPNN(args).to(args.device)
    encoder_loss = ContrastiveLoss(args).to(args.device)
    encoder_optimizer = build_optimizer(encoder, args.cl_init_lr)
    encoder_scheduler = build_lr_scheduler(encoder_optimizer, cl_data_info, args.cl_init_lr, args.cl_max_lr, args.cl_final_lr, args)
    
    metric_func = get_metric_func(args.metric)

    for idx in range(1000):
        if idx == 0:
            for i in range(data_num):
                evaluate_encoder(-1, encoder.encoder, data_info_list[i], train_dataloader_list[i], val_dataloader_list[i], metric_func, 5, args)

        loss = train_encoder(idx, encoder, encoder_loss, encoder_optimizer, encoder_scheduler, cl_data_info, cl_dataloader, args)
        logger.info(f'Epoch {idx}\'s Contrastive Loss: {loss}' )
        for i in range(data_num):
            evaluate_encoder(idx, encoder.encoder, data_info_list[i], train_dataloader_list[i], val_dataloader_list[i], metric_func, 5, args)

    



if __name__ == '__main__':
    main()
