import os
import time
from model import Seq2Seq_bert
from data import Seq2Seq_DataSet
from tools import logging
from config import Config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import BertTokenizer


def train_Seq2Seq(train_data, test_data, model, criterion, optimizer):
    best_accuracy = 0.0
    for epoch in range(Config.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train Seq2Seq model')
        model.train()
        loss_mean = 0.0
        for x, x_mask, y, _ in tqdm(train_data):
            x, x_mask, y = x.to(Config.train_device), x_mask.to(
                Config.train_device), y.to(Config.train_device)
            logits = model(x, x_mask, is_noise=True)
            optimizer.zero_grad()
            logits = logits.reshape(-1, logits.shape[-1])
            y = y.reshape(-1)
            loss = criterion(logits, y)
            loss_mean += loss.item()
            loss.backward()
            optimizer.step()

        loss_mean /= len(train_data)
        print(f"epoch {epoch} train_loss is {loss_mean}")
        eval_accuracy = eval_Seq2Seq(test_data, model)
        print(f"epoch {epoch} test_acc is {eval_accuracy}")
        if best_accuracy < eval_accuracy:
            best_accuracy = eval_accuracy
            logging('Saveing Seq2Seq models...')
            torch.save(model.state_dict(),
                       cur_dir + '/models/Seq2Seq_model_bert.pt')
        if loss_mean < 0.1:
            break


def eval_Seq2Seq(test_data, model):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.eval()
        acc_sum = 0
        n = 0
        for x, x_mask, y, _ in test_data:
            x, x_mask, y = x.to(Config.train_device), x_mask.to(
                Config.train_device), y.to(Config.train_device)
            logits = model(x, x_mask, is_noise=False)
            outputs_idx = logits.argmax(dim=2)
            acc_sum += (outputs_idx == y).float().sum().item()
            n += y.shape[0] * y.shape[1]
            print('-' * Config.sen_len)
            for i in range(len(y)):
                print(' '.join(tokenizer.convert_ids_to_tokens(
                    outputs_idx[i])))
                print(' '.join(tokenizer.convert_ids_to_tokens(y[i])))

            print('-' * Config.sen_len)
        return acc_sum / n


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + Config.train_device.type)
    cur_dir = Config.output_dir + '/Seq2Seq_model/' + str(int(time.time()))
    # make output directory if it doesn't already exist
    if not os.path.isdir(Config.output_dir + '/Seq2Seq_model'):
        os.makedirs(Config.output_dir + '/Seq2Seq_model')
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_dir + '/models')
    logging('Saving into directory' + cur_dir)

    train_dataset_orig = Seq2Seq_DataSet(Config.train_data_path)
    test_dataset_orig = Seq2Seq_DataSet(Config.test_data_path)

    train_data = DataLoader(train_dataset_orig,
                            batch_size=Config.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Config.batch_size,
                           shuffle=False,
                           num_workers=4)

    Seq2Seq_model_bert = Seq2Seq_bert(hidden_size=Config.hidden_size).to(
        Config.train_device)
    logging('Training Seq2Seq Model...')
    criterion_Seq2Seq_model = nn.CrossEntropyLoss().to(Config.train_device)
    optimizer_Seq2Seq_model = optim.Adam(Seq2Seq_model_bert.parameters(),
                                         lr=Config.Seq2Seq_learning_rate)
    train_Seq2Seq(train_data, test_data, Seq2Seq_model_bert,
                  criterion_Seq2Seq_model, optimizer_Seq2Seq_model)
