import os
import time
from baseline_model import Baseline_Model_Bert_Classification, Baseline_Model_LSTM_Classification, Baseline_Model_CNN_Classification
from data import AGNEWS_Dataset, IMDB_Dataset, SNLI_Dataset
from tools import logging
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from config import dataset_config_data, baseline_model_config_data, config_path
from config import AllConfig
from shutil import copyfile


def train_bert_baseline_classifier(model, train_data, test_data,
                                   criterion_baseline_model,
                                   optimizer_baseline_model, cur_dir):
    best_accuracy = 0.0
    for epoch in range(AllConfig.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train baseline_model')
        model.train()
        loss_mean = 0.0
        for x, x_mask, _, label in train_data:
            x, x_mask, label = x.to(AllConfig.train_device), x_mask.to(
                AllConfig.train_device), label.to(AllConfig.train_device)
            logits = model(x, x_mask)
            optimizer_baseline_model.zero_grad()
            loss = criterion_baseline_model(logits, label)
            loss_mean += loss.item()
            loss.backward()
            optimizer_baseline_model.step()

        loss_mean /= len(train_data)
        print(f"epoch {epoch} train_loss is {loss_mean}")
        eval_accuracy = eval_bert_baseline_classifier(model, test_data)
        print(f"epoch {epoch} test_acc is {eval_accuracy}")
        if best_accuracy < eval_accuracy:
            best_accuracy = eval_accuracy
            logging('Saveing baseline models...')
            torch.save(model.state_dict(), cur_dir + r'/baseline_model.pt')
        if loss_mean < 0.1:
            break


def eval_bert_baseline_classifier(model, test_data):
    with torch.no_grad():
        model.eval()
        acc_sum = 0
        n = 0
        for x, x_mask, y in test_data:
            x, x_mask, y = x.to(AllConfig.train_device), x_mask.to(
                AllConfig.train_device), y.to(AllConfig.train_device)
            logits = model(x, x_mask)
            acc_sum += (logits.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n


def train_baseline_classifier(model, train_data, test_data,
                              criterion_baseline_model,
                              optimizer_baseline_model, cur_dir):
    best_accuracy = 0.0
    for epoch in range(AllConfig.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train baseline_model')
        model.train()
        loss_mean = 0.0
        for x, _, _, label in train_data:
            x, label = x.to(AllConfig.train_device), label.to(
                AllConfig.train_device)
            logits = model(x)
            optimizer_baseline_model.zero_grad()
            loss = criterion_baseline_model(logits, label)
            loss_mean += loss.item()
            loss.backward()
            optimizer_baseline_model.step()

        loss_mean /= len(train_data)
        print(f"epoch {epoch} train_loss is {loss_mean}")
        eval_accuracy = eval_baseline_classifier(model, test_data)
        print(f"epoch {epoch} test_acc is {eval_accuracy}")
        if best_accuracy < eval_accuracy:
            best_accuracy = eval_accuracy
            logging('Saveing baseline models...')
            torch.save(model.state_dict(), cur_dir + r'/baseline_model.pt')
        if loss_mean < 0.1:
            break


def eval_baseline_classifier(model, test_data):
    with torch.no_grad():
        model.eval()
        acc_sum = 0
        n = 0
        for x, _, y in test_data:
            x, y = x.to(AllConfig.train_device), y.to(AllConfig.train_device)
            logits = model(x)
            acc_sum += (logits.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n


def build_model():
    if AllConfig.dataset == 'SNLI':
        assert AllConfig.baseline_model == 'BERT' or AllConfig.baseline_model == 'LSTM'
        # TODO: SNLI_baseline_model

    elif AllConfig.dataset == 'AGNEWS':
        if AllConfig.baseline_model == 'BERT':
            baseline_model = Baseline_Model_Bert_Classification(
                dataset_config_data['AGNEWS'])
        elif AllConfig.baseline_model == 'LSTM':
            baseline_model = Baseline_Model_LSTM_Classification(
                dataset_config_data['AGNEWS'], bidirectional=False)
        elif AllConfig.baseline_model == 'BidLSTM':
            baseline_model = Baseline_Model_LSTM_Classification(
                dataset_config_data['AGNEWS'], bidirectional=True)
        elif AllConfig.baseline_model == 'CNN':
            baseline_model = Baseline_Model_CNN_Classification(
                dataset_config_data['AGNEWS'])

    elif AllConfig.dataset == 'IMDB':
        if AllConfig.baseline_model == 'BERT':
            baseline_model = Baseline_Model_Bert_Classification(
                dataset_config_data['IMDB'])
        elif AllConfig.baseline_model == 'LSTM':
            baseline_model = Baseline_Model_LSTM_Classification(
                dataset_config_data['IMDB'], bidirectional=False)
        elif AllConfig.baseline_model == 'BidLSTM':
            baseline_model = Baseline_Model_LSTM_Classification(
                dataset_config_data['IMDB'], bidirectional=True)
        elif AllConfig.baseline_model == 'CNN':
            baseline_model = Baseline_Model_CNN_Classification(
                dataset_config_data['IMDB'])
    return baseline_model


def build_dataset():
    if AllConfig.dataset == 'SNLI':
        train_dataset_orig = SNLI_Dataset(train_data=True,
                                          debug_mode=AllConfig.debug_mode)
        test_dataset_orig = SNLI_Dataset(train_data=False,
                                         debug_mode=AllConfig.debug_mode)
    elif AllConfig.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(train_data=True,
                                            debug_mode=AllConfig.debug_mode)
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           debug_mode=AllConfig.debug_mode)
    elif AllConfig.dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          debug_mode=AllConfig.debug_mode)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         debug_mode=AllConfig.debug_mode)
    train_data = DataLoader(train_dataset_orig,
                            batch_size=AllConfig.batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=AllConfig.batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


def save_config(path):
    copyfile(config_path, path + r'/config.txt')


def train_and_evaluate(model, train_data, test_data, criterion_baseline_model,
                       optimizer_baseline_model, cur_dir):
    if AllConfig.dataset == 'SNLI':
        assert AllConfig.baseline_model == 'BERT' or AllConfig.baseline_model == 'LSTM'
        # TODO: SNLI_baseline_model

    else:
        if AllConfig.baseline_model == 'BERT':
            train_bert_baseline_classifier(model, train_data, test_data,
                                           criterion_baseline_model,
                                           optimizer_baseline_model, cur_dir)
        else:
            train_baseline_classifier(model, train_data, test_data,
                                      criterion_baseline_model,
                                      optimizer_baseline_model, cur_dir)


if __name__ == "__main__":

    logging('Using cuda device gpu: ' + str(AllConfig.cuda_idx))
    cur_dir = AllConfig.output_dir + '/baseline_model/' + AllConfig.dataset + '/' + AllConfig.baseline_model + '/' + str(
        int(time.time()))
    # make output directory if it doesn't already exist
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
    logging('Saving into directory' + cur_dir)
    save_config(cur_dir)

    train_data, test_data = build_dataset()
    model = build_model().to(AllConfig.train_device)

    logging('Training Baseline Model...')
    criterion_baseline_model = nn.CrossEntropyLoss().to(AllConfig.train_device)
    optimizer_baseline_model = optim.Adam(
        model.parameters(),
        lr=baseline_model_config_data[AllConfig.baseline_model].learning_rate[
            AllConfig.dataset])
    train_and_evaluate(model, train_data, test_data, criterion_baseline_model,
                       optimizer_baseline_model, cur_dir)
