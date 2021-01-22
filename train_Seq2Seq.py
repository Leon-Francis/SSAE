import os
import time
from model import Seq2Seq_bert
from data import AGNEWS_Dataset, IMDB_Dataset, SNLI_Dataset
from tools import logging
from config import AttackConfig
from config import config_path
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertAdam
from shutil import copyfile


def train_Seq2Seq(train_data, test_data, model, criterion, optimizer, cur_dir):
    best_accuracy = 0.0
    for epoch in range(AttackConfig.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train Seq2Seq model')
        model.train()
        loss_mean = 0.0
        n = 0
        for x, x_mask, y, _ in train_data:
            x, x_mask, y = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), y.to(AttackConfig.train_device)
            model.zero_grad()
            logits = model(x, x_mask, is_noise=False)
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            loss = criterion(logits, y)
            loss_mean += loss.item()
            loss.backward()
            optimizer.step()
            n += (x.shape[0] * x.shape[1])
        logging(f"epoch {epoch} train_loss is {loss_mean / n}")
        eval_accuracy = evaluate_Seq2Seq(
            test_data, model, cur_dir + f'/eval_Seq2Seq_model_epoch_{epoch}')
        logging(f"epoch {epoch} test_acc is {eval_accuracy}")
        if best_accuracy < eval_accuracy:
            best_accuracy = eval_accuracy
            logging('Saveing Seq2Seq models...')
            torch.save(model.state_dict(), cur_dir + r'/Seq2Seq_model.pt')
        if loss_mean < 0.1:
            break


def evaluate_Seq2Seq(test_data, Seq2Seq_model, path):
    Seq2Seq_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging(f'Saving evaluate of Seq2Seq_model outputs into {path}')
    with torch.no_grad():
        acc_sum = 0
        n = 0
        for x, x_mask, y, _ in test_data:
            x, x_mask, y = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), y.to(AttackConfig.train_device)
            logits = Seq2Seq_model(x, x_mask, is_noise=False)
            # outputs_idx: [batch, sen_len]
            outputs_idx = logits.argmax(dim=2)
            acc_sum += (outputs_idx == y).float().sum().item()
            n += y.shape[0] * y.shape[1]
            with open(path, 'w') as f:
                for i in range(len(y)):
                    f.write('-------orginal sentence----------\n')
                    f.write(' '.join(tokenizer.convert_ids_to_tokens(y[i])) +
                            '\n')
                    f.write(
                        '-------sentence -> encoder -> decoder----------\n')
                    f.write(' '.join(
                        tokenizer.convert_ids_to_tokens(outputs_idx[i])) +
                            '\n' * 2)

    return acc_sum / n


def build_dataset():
    if AttackConfig.dataset == 'SNLI':
        train_dataset_orig = SNLI_Dataset(train_data=True,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = SNLI_Dataset(train_data=False,
                                         debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(train_data=True,
                                            debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         debug_mode=AttackConfig.debug_mode)
    train_data = DataLoader(train_dataset_orig,
                            batch_size=AttackConfig.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=AttackConfig.batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


def save_config(path):
    copyfile(config_path, path + r'/config.txt')


if __name__ == '__main__':
    if AttackConfig.train_multi_cuda:
        logging('Using cuda device gpu: ' + str(AttackConfig.multi_cuda_idx))
    else:
        logging('Using cuda device gpu: ' + str(AttackConfig.cuda_idx))
    cur_dir = AttackConfig.output_dir + '/seq2seq_model/' + AttackConfig.dataset + '/' + str(
        int(time.time()))
    # make output directory if it doesn't already exist
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
    logging('Saving into directory' + cur_dir)
    save_config(cur_dir)

    train_data, test_data = build_dataset()

    model = Seq2Seq_bert().to(AttackConfig.train_device)
    if AttackConfig.train_multi_cuda:
        model = nn.DataParallel(model, device_ids=AttackConfig.multi_cuda_idx)

    logging('Training Seq2Seq Model...')
    criterion_Seq2Seq_model = nn.CrossEntropyLoss().to(
        AttackConfig.train_device)
    if AttackConfig.fine_tuning:
        optimizer_Seq2Seq_model = BertAdam(
            model.parameters(),
            lr=AttackConfig.Seq2Seq_learning_rate,
            warmup=AttackConfig.warmup,
            t_total=len(train_data) * AttackConfig.epochs)
    else:
        optimizer_Seq2Seq_model = optim.AdamW(
            model.parameters(), lr=AttackConfig.Seq2Seq_learning_rate)
    train_Seq2Seq(train_data, test_data, model, criterion_Seq2Seq_model,
                  optimizer_Seq2Seq_model, cur_dir)
