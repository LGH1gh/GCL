import torch
from tqdm import tqdm
import numpy as np

from utils import parse_argument, set_seed, build_lr_scheduler, build_optimizer, get_metric_func, get_loss_func, get_loss_func
from data import load_dataloader, load_cl_dataloader
from model import CMPNNEncoder, FFN4Test, ContrastiveLoss
import logging
logger = logging.getLogger()

def train_encoder(epoch_idx, encoder, criterion, encoder_optimizer, encoder_scheduler, dataloader):
    encoder.train()
    losses = []
    with tqdm(enumerate(dataloader), desc=f'Epoch {epoch_idx}', total=len(dataloader)) as batch:
        for idx, (x, y) in batch:
            encoder_optimizer.zero_grad()
            y_hat = encoder(x)
            loss = criterion(y_hat)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), 0.1)
            batch.set_postfix(loss=loss.item())
            encoder_optimizer.step()
            encoder_scheduler.step()
            losses.append(loss.item())
    return np.mean(losses)

def train_classifer(encoder, classifier, dataloader, classifier_optimizer, classifier_schedule, criterion, args):
    encoder.eval()
    classifier.train()
    with tqdm(enumerate(dataloader), desc='Classifier', total=args.train_steps_per_epoch) as batch:
        for idx, (x, y) in batch:
            y = y.to(args.device)
            classifier_optimizer.zero_grad()
            with torch.no_grad():
                x_hat = encoder(x)
            y_hat = classifier(x_hat)
            loss = criterion(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 0.1)
            batch.set_postfix(loss=loss.item())
            classifier_optimizer.step()
            classifier_schedule.step()

def evaluate_concat(encoder, classifier, dataloader, metric_func, args):
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        target, pred = [], []
        for idx, (x, y) in enumerate(dataloader):
            y_hat = classifier(encoder(x)).reshape(y.size())
            pred.append(y_hat)
            target.append(y)
        pred = torch.cat(pred).cpu()
        target = torch.cat(target).cpu()
        return metric_func(target, pred)


def evaluate_encoder(epoch_idx, encoder, train_dataloader, val_dataloader, metric_func, max_patience, args):
    current_patience, best_result = 0, 0 
    classifier = FFN4Test(args).to(args.device)
    classifier_loss = get_loss_func(args).to(args.device)
    classifier_optimizer = build_optimizer(classifier, args.init_lr)
    classifier_scheduler = build_lr_scheduler(classifier_optimizer, args.train_steps_per_epoch, args.init_lr, args.max_lr, args.final_lr, args)
    while True:
        train_classifer(encoder, classifier, train_dataloader, classifier_optimizer, classifier_scheduler, classifier_loss, args)
        result = evaluate_concat(encoder, classifier, val_dataloader, metric_func, args)
        if best_result < result:
            best_result = result
            current_patience = 0
            print(f'{args.metric} score: {best_result}')
        else:
            current_patience += 1
        if current_patience == max_patience:
            break
    print(f'{args.metric} score: {best_result}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)
    args = parse_argument()
    set_seed(args.seed)
    args.device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
    
    train_dataloader, test_dataloader, val_dataloader = load_dataloader(args)
    # cl_dataloader = load_cl_dataloader(args)


    encoder = CMPNNEncoder(args).to(args.device)
    # cl_loss = ContrastiveLoss(args).to(args.device)
    # encoder_optimizer = build_optimizer(encoder, args.cl_init_lr)
    # encoder_scheduler = build_lr_scheduler(encoder_optimizer, args.cl_steps_per_epoch, args.cl_init_lr, args.cl_max_lr, args.cl_final_lr, args)
    metric_func = get_metric_func(args.metric)
    for idx in range(100):
        if idx == 0:
            evaluate_encoder(idx, encoder, train_dataloader, val_dataloader, metric_func, 5, args)

        # loss = train_encoder(idx, encoder, cl_loss, encoder_optimizer, encoder_scheduler, cl_dataloader)
        # if loss < 0.4 and idx % 5 == 0:
        #     evaluate_encoder(idx, encoder, train_dataloader, val_dataloader, metric_func, 5, args)