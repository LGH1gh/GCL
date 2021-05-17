import logging
import torch
import numpy as np
from utils import parse_argument, set_seed, build_lr_scheduler, build_optimizer, get_metric_func, get_loss_func
from data import load_dataloader
from model import CMPNN, FFN4Test


logger = logging.getLogger()


class Trainer:
    def __init__(self):

        logging.basicConfig(format='%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
        logger.info('#'*100)
        args = parse_argument()
        set_seed(args.seed)
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.metric_func = get_metric_func(args.metric)


        self.train_dataloader_list, self.val_dataloader_list, self.test_dataloader_list, self.data_info_list = [], [], [], []
        for name in args.data_name:
            train_dataloader, val_dataloader, test_dataloader, data_info = load_dataloader(args, name)
            self.train_dataloader_list.append(train_dataloader)
            self.val_dataloader_list.append(val_dataloader)
            self.test_dataloader_list.append(test_dataloader)
            self.data_info_list.append(data_info)
        self.data_num = len(args.data_name)

        self.encoder_list, self.encoder_optimizer_list, self.encoder_scheduler_list, self.classifier_list, self.classifier_optimizer_list, self.classifier_scheduler_list, self.loss_func_list,  = [], [], [], [], [], [], []

        for idx in range(self.data_num):
            
            self.encoder_list.append(CMPNN(args).encoder.to(self.device))
            self.encoder_optimizer_list.append(build_optimizer(self.encoder_list[-1], args.init_lr))
            self.encoder_scheduler_list.append(build_lr_scheduler(self.encoder_optimizer_list[-1], self.data_info_list[idx], args.init_lr, args.max_lr, args.final_lr, args))
            self.classifier_list.append(FFN4Test(args, self.data_info_list[idx]).to(self.device))
            self.classifier_optimizer_list.append(build_optimizer(self.classifier_list[-1], args.init_lr))
            self.classifier_scheduler_list.append(build_lr_scheduler(self.classifier_optimizer_list[-1], self.data_info_list[idx], args.init_lr, args.max_lr, args.final_lr, args))
            self.loss_func_list.append(get_loss_func(self.data_info_list[idx]).to(self.device))
            


    def run_train_epoch(self, epoch, encoder, encoder_optimizer, encoder_scheduler, classifier, classifier_optimizer, classifier_scheduler, dataloader, data_info, criterion):
        encoder.train()
        classifier.train()
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            mask = torch.Tensor([[not val.isnan() for val in col] for col in y]).to(self.device)
            y = torch.Tensor([[0 if val.isnan() else val for val in col] for col in y]).to(self.device)
            y_hat = classifier(encoder(x, x.ndata['attr'], x.edata['attr']))
            # Mask non-existing labels
            loss = criterion(y_hat, y)
            print(criterion)
            print(y_hat)
            encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            loss = torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)
            print(loss)
            exit()
            loss = loss.sum() / mask.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.1)
            encoder_optimizer.step()
            encoder_scheduler.step()
            classifier_optimizer.step()
            classifier_scheduler.step()

    def run_eval_epoch(self, epoch, encoder, classifier, dataloader, data_info, metric_func):
        encoder.eval()
        classifier.eval()
        with torch.no_grad():
            target, pred = [], []
            for idx, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
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

    def run(self):
        for epoch in range(100):
            for idx in range(self.data_num):
                self.run_train_epoch(epoch, self.encoder_list[idx], self.encoder_optimizer_list[idx], self.encoder_scheduler_list[idx], self.classifier_list[idx], self.classifier_optimizer_list[idx], self.classifier_scheduler_list[idx], self.train_dataloader_list[idx], self.data_info_list[idx], self.loss_func_list[idx])
            # Validation and early stop
                val_score = self.run_eval_epoch(epoch, self.encoder_list[idx], self.classifier_list[idx], self.val_dataloader_list[idx], self.data_info_list[idx], self.metric_func)
                print(f'Epoch {epoch} {self.data_info_list[idx]["data_name"]}\'s score: {val_score}')



if __name__ == '__main__':
    from argparse import ArgumentParser
    trainer = Trainer()
    trainer.run()
