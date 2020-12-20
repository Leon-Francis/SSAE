import os
import time
from baseline_model import Baseline_Model_Bert
from data import Baseline_Dataset
from tools import logging
from config import Config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


def train_baseline():
    best_accuracy = 0.0
    for epoch in range(Config.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train baseline_model')
        baseline_model_bert.train()
        loss_mean = 0.0
        for x, x_mask, y in tqdm(train_data):
            x, x_mask, y = x.to(Config.train_device), x_mask.to(
                Config.train_device), y.to(Config.train_device)
            logits = baseline_model_bert(x, x_mask)
            optimizer_baseline_model.zero_grad()
            loss = criterion_baseline_model(logits, y)
            loss_mean += loss.item()
            loss.backward()
            optimizer_baseline_model.step()

        loss_mean /= len(train_data)
        print(f"epoch {epoch} train_loss is {loss_mean}")
        eval_accuracy = eval_baseline()
        print(f"epoch {epoch} test_acc is {eval_accuracy}")
        if best_accuracy < eval_accuracy:
            best_accuracy = eval_accuracy
            logging('Saveing baseline models...')
            torch.save(baseline_model_bert.state_dict(),
                       cur_dir + '/models/baseline_model_bert.pt')
        if loss_mean < 0.1:
            break


def eval_baseline():
    with torch.no_grad():
        baseline_model_bert.eval()
        acc_sum = 0
        n = 0
        for x, x_mask, y in test_data:
            x, x_mask, y = x.to(Config.train_device), x_mask.to(
                Config.train_device), y.to(Config.train_device)
            logits = baseline_model_bert(x, x_mask)
            acc_sum += (logits.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n


if __name__ == "__main__":
    logging('Using cuda device gpu: ' + Config.train_device.type)
    cur_dir = Config.output_dir + '/baseline_model/' + str(int(time.time()))
    # make output directory if it doesn't already exist
    if not os.path.isdir(Config.output_dir + '/baseline_model'):
        os.makedirs(Config.output_dir + '/baseline_model')
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_dir + '/models')
    logging('Saving into directory' + cur_dir)

    train_dataset_orig = Baseline_Dataset(Config.train_data_path)
    test_dataset_orig = Baseline_Dataset(Config.test_data_path)

    train_data = DataLoader(train_dataset_orig,
                            batch_size=Config.batch_size,
                            shuffle=True)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=Config.batch_size,
                           shuffle=False)

    baseline_model_bert = Baseline_Model_Bert().to(Config.train_device)
    logging('Training Baseline Model...')
    criterion_baseline_model = nn.CrossEntropyLoss().to(Config.train_device)
    optimizer_baseline_model = optim.Adam(baseline_model_bert.parameters(),
                                          lr=Config.baseline_train_rate)
    train_baseline()
