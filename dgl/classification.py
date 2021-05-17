import logging
import torch
import numpy as np
from utils import parse_argument, set_seed, build_lr_scheduler, build_optimizer, get_metric_func, get_loss_func
from data import load_dataloader
from model import CMPNN


logger = logging.getLogger()

class Trainer:
    def __init__(self):
        # device
        logging.basicConfig(format='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
        logger.info('#'*100)
        args = parse_argument()
        set_seed(args.seed)
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.metric_func = get_metric_func(args.metric)
        self.train_dataloader, self.val_dataloader, self.test_dataloader, self.data_info = load_dataloader(args, 'bbbp')
        self.model = CMPNN(args).to(self.device)
        self.optimizer = build_optimizer(self.model, args.init_lr)
        self.scheduler = build_lr_scheduler(self.optimizer, self.data_info, args.init_lr, args.max_lr, args.final_lr, args)
        self.criterion = get_loss_func(self.data_info).to(self.device)
            
    
    def run_train_epoch(self):
        self.model.train()
        for batch_id, (batch_graph, labels) in enumerate(self.train_dataloader):
            batch_graph = batch_graph.to(self.device)
            mask = torch.Tensor([[not val.isnan() for val in col] for col in labels]).to(self.device)
            labels = torch.Tensor([[0 if val.isnan() else val for val in col] for col in labels]).to(self.device)

            logits = self.model(batch_graph, batch_graph.ndata['attr'], batch_graph.edata['attr'])
            # Mask non-existing labels
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
            loss = loss.sum() / mask.sum()
            loss.backward()
            print(loss.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

    def run_eval_epoch(self):
        self.model.eval()
        with torch.no_grad():
            target, pred = [], []
            for batch_id, (batch_graph, labels) in enumerate(self.val_dataloader):
                batch_graph = batch_graph.to(self.device)
                logits = self.model(batch_graph)
                pred.append(logits.cpu())
                target.append(labels)

            pred = torch.cat(pred)
            target = torch.cat(target)
            valid_preds = [[] for _ in range(self.data_info['task_num'])]
            valid_targets = [[] for _ in range(self.data_info['task_num'])]
            for i in range(self.data_info['task_num']):
                for j in range(len(pred)):
                    if not target[j][i].isnan():  # Skip those without targets
                        valid_preds[i].append(pred[j][i])
                        valid_targets[i].append(target[j][i])
            
            results = []
            for i in range(self.data_info['task_num']):
                if self.data_info['task_type'] == 'classification':
                    results.append(self.metric_func(valid_targets[i], valid_preds[i]))
        return np.nanmean(results)

    def run(self):
        for epoch in range(100):
            self.run_train_epoch()
            val_score = self.run_eval_epoch()
            print(f'Epoch {epoch} {self.data_info["data_name"]}\'s score: {val_score}')



if __name__ == '__main__':
    from argparse import ArgumentParser
    trainer = Trainer()
    trainer.run()
