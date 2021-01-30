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
from shutil import copyfile
from baseline_module.baseline_model_builder import BaselineModelBuilder
from huffman_tree import HuffmanTree


def train_Seq2Seq(train_data, test_data, model, huffman_tree, criterion,
                  optimizer_Seq2Seq, optimizer_ht, cur_dir, attack_vocab):
    best_accuracy = 0.0
    for epoch in range(AttackConfig.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train Seq2Seq model')
        model.train()
        huffman_tree.train()
        loss_mean = 0.0
        n = 0
        for x, x_mask, y, _ in train_data:
            x, x_mask, y = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), y.to(AttackConfig.train_device)
            model.zero_grad()
            huffman_tree.zero_grad()
            logits = model(x, x_mask, is_noise=False)
            loss = huffman_tree(logits, y)
            loss_mean += loss.item()
            loss.backward()
            optimizer_Seq2Seq.step()
            optimizer_ht.step()
            n += (x.shape[0] * x.shape[1])
        logging(f"epoch {epoch} train_loss is {loss_mean / n}")
        eval_accuracy = evaluate_Seq2Seq(
            test_data, model, huffman_tree,
            cur_dir + f'/eval_Seq2Seq_model_epoch_{epoch}', attack_vocab)
        logging(f"epoch {epoch} test_acc is {eval_accuracy}")
        if best_accuracy < eval_accuracy:
            best_accuracy = eval_accuracy
            logging('Saveing Seq2Seq models...')
            torch.save(model.state_dict(), cur_dir + r'/Seq2Seq_model.pt')


def evaluate_Seq2Seq(test_data, Seq2Seq_model, huffman_tree, path,
                     attack_vocab):
    Seq2Seq_model.eval()
    huffman_tree.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging(f'Saving evaluate of Seq2Seq_model outputs into {path}')
    with torch.no_grad():
        acc_sum = 0
        n = 0
        for x, x_mask, y, _ in test_data:
            x, x_mask, y = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), y.to(AttackConfig.train_device)
            # outputs_idx: [batch, sen_len]
            outputs_idx = huffman_tree.get_sentence(
                Seq2Seq_model(x, x_mask, is_noise=False))
            acc_sum += (outputs_idx == y).float().sum().item()
            n += y.shape[0] * y.shape[1]
            if attack_vocab:
                with open(path, 'a') as f:
                    for i in range(len(y)):
                        f.write('-------orginal sentence----------\n')
                        f.write(' '.join(
                            [attack_vocab.get_word(token)
                             for token in y[i]]) + '\n')
                        f.write(
                            '-------sentence -> encoder -> decoder----------\n'
                        )
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in outputs_idx[i]
                        ]) + '\n' * 2)
            else:
                with open(path, 'a') as f:
                    for i in range(len(y)):
                        f.write('-------orginal sentence----------\n')
                        f.write(
                            ' '.join(tokenizer.convert_ids_to_tokens(y[i])) +
                            '\n')
                        f.write(
                            '-------sentence -> encoder -> decoder----------\n'
                        )
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(outputs_idx[i])) +
                                '\n' * 2)

        return acc_sum / n


def build_dataset(attack_vocab):
    if AttackConfig.dataset == 'SNLI':
        train_dataset_orig = SNLI_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = SNLI_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(
            train_data=True,
            attack_vocab=attack_vocab,
            debug_mode=AttackConfig.debug_mode,
        )
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           attack_vocab=attack_vocab,
                                           debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
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

    baseline_model_builder = BaselineModelBuilder(AttackConfig.dataset,
                                                  'LSTM',
                                                  AttackConfig.train_device,
                                                  is_load=True)

    word_count = {
        k: v
        for k, v in sorted(baseline_model_builder.vocab.word_count.items(),
                           key=lambda x: x[1],
                           reverse=True)
    }

    train_data, test_data = build_dataset(
        attack_vocab=baseline_model_builder.vocab)

    model = Seq2Seq_bert(baseline_model_builder.vocab.num).to(
        AttackConfig.train_device)
    huffman_tree = HuffmanTree(word_count, baseline_model_builder.vocab).to(
        AttackConfig.train_device)
    if AttackConfig.train_multi_cuda:
        model = nn.DataParallel(model, device_ids=AttackConfig.multi_cuda_idx)

    logging('Training Seq2Seq Model...')
    criterion_Seq2Seq_model = nn.CrossEntropyLoss().to(
        AttackConfig.train_device)
    if AttackConfig.fine_tuning:
        optimizer_Seq2Seq_model = optim.AdamW(
            [{
                'params': model.encoder.parameters(),
                'lr': AttackConfig.Seq2Seq_learning_rate_BERT
            }, {
                'params': model.decoder.parameters()
            }],
            lr=AttackConfig.Seq2Seq_learning_rate)
    else:
        optimizer_Seq2Seq_model = optim.AdamW(
            model.parameters(), lr=AttackConfig.Seq2Seq_learning_rate)
    optimizer_ht = optim.AdamW(huffman_tree.parameters(),
                               lr=AttackConfig.Seq2Seq_learning_rate)
    train_Seq2Seq(train_data, test_data, model, huffman_tree,
                  criterion_Seq2Seq_model, optimizer_Seq2Seq_model,
                  optimizer_ht, cur_dir, baseline_model_builder.vocab)
