import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from perturb import perturb
from shutil import copyfile
from model import Seq2Seq_bert, LSTM_G, LSTM_A
from data import AGNEWS_Dataset, IMDB_Dataset, SNLI_Dataset
from tools import logging
from config import config_path, AttackConfig
from baseline_module.baseline_model_builder import BaselineModelBuilder


def train_Seq2Seq(train_data, model, criterion, optimizer, total_loss):
    model.train()
    x, x_mask, _, x_label, _, _, _, _, _, _ = train_data
    logits = model(x, x_mask, is_noise=False)
    model.zero_grad()
    logits = logits.reshape(-1, logits.shape[-1])
    x_label = x_label.reshape(-1)
    loss = criterion(logits, x_label)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss


def train_gan_a(train_data, Seq2Seq_model, gan_gen, gan_adv, baseline_model,
                optimizer_gan_g, optimizer_gan_a, criterion_ce):
    gan_gen.train()
    gan_adv.train()
    baseline_model.train()
    optimizer_gan_a.zero_grad()
    optimizer_gan_g.zero_grad()

    x, x_mask, x_len, _, _, y_mask, y_len, y_label, whole_type, label = train_data
    # perturb_x: [batch, sen_len]
    perturb_x = Seq2Seq_model(x,
                              x_mask,
                              is_noise=False,
                              generator=gan_gen,
                              adversary=gan_adv).argmax(dim=2)
    if AttackConfig.baseline_model in ['Bert', 'Bert_E']:
        # perturb_x_mask: [batch, seq_len]
        perturb_x_mask = torch.ones(perturb_x.shape, dtype=torch.int64)
        with torch.no_grad():
            # mask before [SEP]
            for i in range(perturb_x.shape[0]):
                for word_idx in range(perturb_x.shape[1]):
                    if perturb_x[i][word_idx].item() == 102:
                        perturb_x_mask[i][word_idx + 1:] = 0
                        break
        perturb_x_mask = perturb_x_mask.to(AttackConfig.train_device)
        # perturb_logits: [batch, 4]
        perturb_logits = baseline_model(
            torch.cat((perturb_x, y_label), dim=1), whole_type,
            torch.cat((perturb_x_mask, y_mask), dim=1))
    else:
        perturb_logits = baseline_model(((perturb_x, x_len), (y_label, y_len)))

    loss = criterion_ce(perturb_logits, label)
    loss *= -5
    loss.backward()
    optimizer_gan_a.step()
    optimizer_gan_g.step()

    return -loss.item()


def train_gan_g(train_data, Seq2Seq_model, gan_gen, gan_adv, criterion_mse,
                optimizer_gan_g, optimizer_gan_a):
    gan_gen.train()
    gan_adv.train()
    optimizer_gan_a.zero_grad()
    optimizer_gan_g.zero_grad()

    x, x_mask, _, x_label, _, _, _, _, _, _ = train_data
    # real_hidden: [batch, sen_len, hidden]
    real_hidden = Seq2Seq_model(x, x_mask, is_noise=False, encode_only=True)
    fake_hidden = gan_gen(gan_adv(real_hidden))

    loss = criterion_mse(real_hidden.reshape(real_hidden.shape[0], -1),
                         fake_hidden.reshape(fake_hidden.shape[0], -1))

    loss.backward()
    optimizer_gan_g.step()
    optimizer_gan_a.step()

    return loss.item()


def evaluate_gan(test_data, Seq2Seq_model, gan_gen, gan_adv, dir,
                 attack_vocab):
    Seq2Seq_model.eval()
    gan_gen.eval()
    gan_adv.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging(f'Saving evaluate of gan outputs into {dir}')
    with torch.no_grad():

        for x, x_mask, _, x_label, _, _, _, _, _, _ in test_data:
            x, x_mask, x_label = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), x_label.to(
                    AttackConfig.train_device)

            # sentence -> encoder -> decoder
            Seq2Seq_outputs = Seq2Seq_model(x, x_mask, is_noise=False)
            # Seq2Seq_idx: [batch, seq_len]
            Seq2Seq_idx = Seq2Seq_outputs.argmax(dim=2)

            # sentence -> encoder -> adversary -> generator ->  decoder
            # eagd_outputs: [batch, seq_len, vocab_size]
            eagd_outputs = Seq2Seq_model(x,
                                         x_mask,
                                         is_noise=False,
                                         generator=gan_gen,
                                         adversary=gan_adv)
            # eagd_idx: [batch_size, sen_len]
            eagd_idx = eagd_outputs.argmax(dim=2)

            if attack_vocab:
                with open(dir, 'a') as f:
                    for i in range(len(x_label)):
                        f.write('------orginal sentence---------\n')
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in x_label[i]
                        ]) + '\n')
                        f.write('------setence -> encoder -> decoder-------\n')
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in Seq2Seq_idx[i]
                        ]) + '\n')
                        f.write(
                            '------sentence -> encoder -> inverter -> generator -> decoder-------\n'
                        )
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in eagd_idx[i]
                        ]) + '\n' * 2)
            else:
                with open(dir, 'a') as f:
                    for i in range(len(x_label)):
                        f.write('------orginal sentence---------\n')
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(x_label[i])) +
                                '\n')
                        f.write('------setence -> encoder -> decoder-------\n')
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(Seq2Seq_idx[i])) +
                                '\n')
                        f.write(
                            '------sentence -> encoder -> inverter -> generator -> decoder-------\n'
                        )
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(eagd_idx[i])) +
                                '\n' * 2)


def evaluate_Seq2Seq(test_data, Seq2Seq_model, dir, attack_vocab):
    Seq2Seq_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging(f'Saving evaluate of Seq2Seq_model outputs into {dir}')
    with torch.no_grad():
        acc_sum = 0
        n = 0
        for x, x_mask, _, x_label, _, _, _, _, _, _ in test_data:
            x, x_mask, x_label = x.to(AttackConfig.train_device), x_mask.to(
                AttackConfig.train_device), x_label.to(
                    AttackConfig.train_device)
            logits = Seq2Seq_model(x, x_mask, is_noise=False)
            # outputs_idx: [batch, sen_len]
            outputs_idx = logits.argmax(dim=2)
            acc_sum += (outputs_idx == x_label).float().sum().item()
            n += x_label.shape[0] * x_label.shape[1]

            if attack_vocab:
                with open(dir, 'a') as f:
                    for i in range(len(x_label)):
                        f.write('-------orginal sentence----------\n')
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in x_label[i]
                        ]) + '\n')
                        f.write(
                            '-------sentence -> encoder -> decoder----------\n'
                        )
                        f.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in outputs_idx[i]
                        ]) + '\n' * 2)
            else:
                with open(dir, 'a') as f:
                    for i in range(len(x_label)):
                        f.write('-------orginal sentence----------\n')
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(x_label[i])) +
                                '\n')
                        f.write(
                            '-------sentence -> encoder -> decoder----------\n'
                        )
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(outputs_idx[i])) +
                                '\n' * 2)

        return acc_sum / n


def save_all_models(Seq2Seq_model, gan_gen, gan_adv, dir):
    logging('Saving models...')
    torch.save(Seq2Seq_model.state_dict(), dir + '/Seq2Seq_model.pt')
    torch.save(gan_gen.state_dict(), dir + '/gan_gen.pt')
    torch.save(gan_adv.state_dict(), dir + '/gan_adv.pt')


def save_config(path):
    copyfile(config_path, path + r'/config.txt')


def build_dataset(attack_vocab):
    if AttackConfig.dataset == 'SNLI':
        train_dataset_orig = SNLI_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=AttackConfig.debug_mode)
        test_dataset_orig = SNLI_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=AttackConfig.debug_mode)
    elif AttackConfig.dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(train_data=True,
                                            attack_vocab=attack_vocab,
                                            debug_mode=AttackConfig.debug_mode)
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


if __name__ == '__main__':
    logging('Using cuda device gpu: ' + str(AttackConfig.cuda_idx))
    cur_dir = AttackConfig.output_dir + '/gan_model/' + AttackConfig.dataset + '/' + AttackConfig.baseline_model + '/' + str(
        int(time.time()))
    cur_dir_models = cur_dir + '/models'
    # make output directory if it doesn't already exist
    if not os.path.isdir(cur_dir):
        os.makedirs(cur_dir)
        os.makedirs(cur_dir_models)
    logging('Saving into directory' + cur_dir)
    save_config(cur_dir)

    baseline_model_builder = BaselineModelBuilder(AttackConfig.dataset,
                                                  AttackConfig.baseline_model,
                                                  AttackConfig.train_device,
                                                  is_load=True)

    # prepare dataset
    logging('preparing data...')
    train_data, test_data = build_dataset(baseline_model_builder.vocab)

    # init models
    logging('init models, optimizer, criterion...')
    Seq2Seq_model = Seq2Seq_bert(
        baseline_model_builder.vocab.num
        if baseline_model_builder.vocab else AttackConfig.vocab_size,
        bidirectional=AttackConfig.Seq2Seq_BidLSTM).to(
            AttackConfig.train_device)
    if AttackConfig.load_pretrained_Seq2Seq:
        Seq2Seq_model.load_state_dict(
            torch.load(AttackConfig.pretrained_Seq2Seq_path,
                       map_location=AttackConfig.train_device))
    gan_gen = LSTM_G(AttackConfig.super_hidden_size,
                     AttackConfig.hidden_size,
                     num_layers=3).to(AttackConfig.train_device)

    gan_adv = LSTM_A(AttackConfig.hidden_size,
                     AttackConfig.super_hidden_size,
                     num_layers=3).to(AttackConfig.train_device)
    baseline_model = baseline_model_builder.net

    # init optimizer
    if AttackConfig.fine_tuning:
        optimizer_Seq2Seq = optim.AdamW(
            [{
                'params': Seq2Seq_model.encoder.parameters(),
                'lr': AttackConfig.Seq2Seq_learning_rate_BERT
            }, {
                'params': Seq2Seq_model.decoder.parameters()
            }, {
                'params': Seq2Seq_model.fc.parameters()
            }],
            lr=AttackConfig.Seq2Seq_learning_rate_LSTM)
    else:
        optimizer_Seq2Seq = optim.AdamW(Seq2Seq_model.parameters(),
                                        lr=AttackConfig.Seq2Seq_learning_rate)
    optimizer_gan_g = optim.AdamW(gan_gen.parameters(),
                                  lr=AttackConfig.gan_gen_learning_rate)
    optimizer_gan_a = optim.AdamW(gan_adv.parameters(),
                                  lr=AttackConfig.gan_adv_learning_rate)
    # init criterion
    criterion_ce = nn.CrossEntropyLoss().to(AttackConfig.train_device)
    criterion_mse = nn.MSELoss().to(AttackConfig.train_device)

    # start training
    logging('Training Seq2Seq Model...')

    niter_gan = 1

    for epoch in range(AttackConfig.epochs):
        if epoch in AttackConfig.gan_schedule:
            niter_gan += 1
        niter = 0
        total_loss_Seq2Seq = 0
        total_loss_gan_a = 0
        total_loss_gan_g = 0
        logging(f'Training {epoch} epoch')
        for x, x_mask, x_len, x_label, y, y_mask, y_len, y_label, whole_type, label in train_data:
            niter += 1
            x, x_mask, x_len, x_label, y, y_mask, y_len, y_label, whole_type, label = x.to(
                AttackConfig.train_device
            ), x_mask.to(AttackConfig.train_device), x_len.to(
                AttackConfig.train_device), x_label.to(
                    AttackConfig.train_device), y.to(
                        AttackConfig.train_device), y_mask.to(
                            AttackConfig.train_device), y_len.to(
                                AttackConfig.train_device), y_label.to(
                                    AttackConfig.train_device), whole_type.to(
                                        AttackConfig.train_device), label.to(
                                            AttackConfig.train_device)

            if not AttackConfig.load_pretrained_Seq2Seq:
                for i in range(AttackConfig.seq2seq_train_times):
                    total_loss_Seq2Seq += train_Seq2Seq(
                        (x, x_mask, x_len, x_label, y, y_mask, y_len, y_label,
                         whole_type, label), Seq2Seq_model, criterion_ce,
                        optimizer_Seq2Seq, total_loss_Seq2Seq)
            else:
                if AttackConfig.fine_tuning:
                    for i in range(AttackConfig.seq2seq_train_times):
                        total_loss_Seq2Seq += train_Seq2Seq(
                            (x, x_mask, x_len, x_label, y, y_mask, y_len,
                             y_label, whole_type, label), Seq2Seq_model,
                            criterion_ce, optimizer_Seq2Seq,
                            total_loss_Seq2Seq)

            for k in range(niter_gan):
                if epoch < AttackConfig.gan_gen_train_limit:
                    for i in range(AttackConfig.gan_gen_train_times):
                        total_loss_gan_g += train_gan_g(
                            (x, x_mask, x_len, x_label, y, y_mask, y_len,
                             y_label, whole_type, label), Seq2Seq_model,
                            gan_gen, gan_adv, criterion_mse, optimizer_gan_g,
                            optimizer_gan_a)

                for i in range(AttackConfig.gan_adv_train_times):
                    total_loss_gan_a += train_gan_a(
                        (x, x_mask, x_len, x_label, y, y_mask, y_len, y_label,
                         whole_type, label), Seq2Seq_model, gan_gen, gan_adv,
                        baseline_model, optimizer_gan_g, optimizer_gan_a,
                        criterion_ce)

            if niter % 100 == 0:
                # decaying noise
                logging(
                    f'epoch {epoch}, niter {niter}:Loss_Seq2Seq: {total_loss_Seq2Seq / niter / AttackConfig.batch_size / AttackConfig.seq2seq_train_times}, Loss_gan_g: {total_loss_gan_g / niter / AttackConfig.batch_size / AttackConfig.gan_gen_train_times}, Loss_gan_a: {total_loss_gan_a / niter / AttackConfig.batch_size / AttackConfig.gan_adv_train_times}'
                )

        # end of epoch --------------------------------
        # evaluation

        logging(f'epoch {epoch} evaluate Seq2Seq model')
        Seq2Seq_acc = evaluate_Seq2Seq(
            test_data, Seq2Seq_model,
            cur_dir_models + f'/epoch{epoch}_evaluate_Seq2Seq',
            baseline_model_builder.vocab)
        logging(f'Seq2Seq_model acc = {Seq2Seq_acc}')

        logging(f'epoch {epoch} evaluate gan')
        evaluate_gan(test_data, Seq2Seq_model, gan_gen, gan_adv,
                     cur_dir_models + f'/epoch{epoch}_evaluate_gan',
                     baseline_model_builder.vocab)

        if (epoch + 1) % 5 == 0:
            os.makedirs(cur_dir_models + f'/epoch{epoch}')
            save_all_models(Seq2Seq_model, gan_gen, gan_adv,
                            cur_dir_models + f'/epoch{epoch}')

            logging(f'epoch {epoch} Staring perturb')
            # attach_acc: [search_time, sample_num]
            attack_acc = perturb(test_data, Seq2Seq_model, gan_gen, gan_adv,
                                 baseline_model,
                                 cur_dir + f'/epoch{epoch}_perturb',
                                 baseline_model_builder.vocab)
            log = ''
            for j in range(AttackConfig.perturb_search_times):
                log += f'search_times {j} attact success acc'
                for i in range(AttackConfig.perturb_sample_num):
                    log += f'{attack_acc[j][i]}' + '\t' * 2
                log += '\n'
            logging(log)
