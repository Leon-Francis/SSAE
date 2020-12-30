import os
import time
from model import Seq2Seq_bert, MLP_G, MLP_D, MLP_I, JSDistance
from data import Seq2Seq_DataSet
from tools import logging
from config import Config
from torch import nn, optim
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer


def train_Seq2Seq(train_data, model, criterion, optimizer, total_loss):
    model.train()
    x, x_mask, y = train_data
    x, x_mask, y = x.to(Config.train_device), x_mask.to(
        Config.train_device), y.to(Config.train_device)
    logits = model(x, x_mask, is_noise=True)
    optimizer.zero_grad()
    logits = logits.reshape(-1, logits.shape[-1])
    y = y.reshape(-1)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    return total_loss


def train_gan_d(train_data, Seq2Seq_model, gan_gen, gan_disc,
                optimizer_Seq2Seq, optimizer_gan_d):
    for p in gan_disc.parameters():
        p.data.clamp_(-0.01, 0.01)
    Seq2Seq_model.train()
    optimizer_Seq2Seq.zero_grad()
    gan_disc.train()
    optimizer_gan_d.zero_grad()
    x, x_mask, y = train_data
    x, x_mask, y = x.to(Config.train_device), x_mask.to(
        Config.train_device), y.to(Config.train_device)

    real_hidden = Seq2Seq_model(x, x_mask, is_noise=False, encode_only=True)

    errD_real = gan_disc(real_hidden)
    errD_real.backward()

    noise = torch.ones(Config.batch_size,
                       Config.super_hidden_size).to(Config.train_device)
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake *= -1.0
    errD_fake.backward()

    nn.utils.clip_grad_norm_(Seq2Seq_model.parameters(), 1)

    optimizer_Seq2Seq.step()
    optimizer_gan_d.step()

    errD = -(errD_real - errD_fake)
    return errD


def train_gan_g(gan_gen, gan_disc, optimizer_gan_g):
    gan_gen.train()
    optimizer_gan_g.zero_grad()

    noise = torch.ones(Config.batch_size,
                       Config.super_hidden_size).to(Config.train_device)
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)

    errG.backward()
    optimizer_gan_g.step()

    return errG


def train_inv(train_data,
              Seq2Seq_model,
              gan_gen,
              inverter,
              optimizer_inv,
              criterion_ce,
              criterion_js,
              criterion_mse,
              gamma=0.5):
    inverter.train()
    optimizer_inv.zero_grad()

    noise = torch.ones(Config.batch_size,
                       Config.super_hidden_size).to(Config.train_device)
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    inv_noise = inverter(fake_hidden)
    errI = criterion_js(inv_noise, noise)

    x, x_mask, y = train_data
    x, x_mask, y = x.to(Config.train_device), x_mask.to(
        Config.train_device), y.to(Config.train_device)

    real_hidden = Seq2Seq_model(x, x_mask, is_noise=True, encode_only=True)

    real_noise = inverter(real_hidden)
    hidden = gan_gen(real_noise)
    errI = gamma * errI
    errI += (1 - gamma) * criterion_mse(hidden, real_hidden)

    torch.nn.utils.clip_grad_norm_(Seq2Seq_model.parameters(), 1)

    # loss / backprop
    errI.backward()
    optimizer_inv.step()

    return errI


def evaluate_generator(noise, Seq2Seq_model, gan_gen):
    gan_gen.eval()
    Seq2Seq_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        fake_hidden = gan_gen(noise)
        # outputs: [batch_size, sen_len, vocab_size]
        outputs = Seq2Seq_model.decode(fake_hidden)
        # outputs_idx: [batch_size, sen_len]
        outputs_idx = outputs.argmax(dim=2)
        logging("generator outputs:")
        for idx in outputs_idx:
            print(' '.join(tokenizer.convert_ids_to_tokens(outputs_idx[idx])))


def eval_Seq2Seq(test_data, model):
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.eval()
        acc_sum = 0
        n = 0
        for x, x_mask, y in test_data:
            x, x_mask, y = x.to(Config.train_device), x_mask.to(
                Config.train_device), y.to(Config.train_device)
            logits = model(x, x_mask, is_noise=False)
            outputs_idx = logits.argmax(dim=2)
            acc_sum += (outputs_idx == y).float().sum().item()
            n += y.shape[0] * y.shape[1]
            print('-' * Config.sen_len)
            for i in range(5):
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

    # prepare dataset
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

    # init models
    Seq2Seq_model_bert = Seq2Seq_bert(hidden_size=Config.hidden_size).to(
        Config.train_device)
    gan_gen = MLP_G(Config.super_hidden_size,
                    Config.hidden_size).to(Config.train_device)
    gan_disc = MLP_D(Config.hidden_size, 1).to(Config.train_device)
    inverter = MLP_I(Config.hidden_size,
                     Config.super_hidden_size).to(Config.train_device)

    # init optimizer
    optimizer_Seq2Seq = optim.Adam(Seq2Seq_model_bert.parameters(),
                                   lr=Config.Seq2Seq_learning_rate)
    optimizer_inv = optim.Adam(inverter.parameters(),
                               lr=Config.inverter_learning_rate,
                               betas=Config.optim_betas)
    optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                                 lr=Config.gan_gen_learning_rate,
                                 betas=Config.optim_betas)
    optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                                 lr=Config.gan_disc_learning_rate,
                                 betas=Config.optim_betas)
    # init criterion
    criterion_ce = nn.CrossEntropyLoss().to(Config.train_device)
    criterion_mse = nn.MSELoss().to(Config.train_device)
    criterion_js = JSDistance().to(Config.train_device)
    criterion_Seq2Seq_model = nn.CrossEntropyLoss().to(Config.train_device)

    # start training
    logging('Training Seq2Seq Model...')

    for epoch in range(Config.epochs):
        niter = 0
        total_loss_Seq2Seq = 0
        total_loss_gan_d = 0
        total_loss_gan_g = 0
        total_loss_inv = 0
        logging(f'Training {epoch} epoch')
        for x, x_mask, y in train_data:
            niter += 1
            total_loss_Seq2Seq += train_Seq2Seq(
                (x, x_mask, y), Seq2Seq_model_bert, criterion_ce,
                optimizer_Seq2Seq, total_loss_Seq2Seq)
            for i in range(5):
                total_loss_gan_d += train_gan_d(
                    (x, x_mask, y), Seq2Seq_model_bert, gan_gen, gan_disc,
                    optimizer_Seq2Seq, optimizer_gan_d)

            total_loss_gan_g += train_gan_g(gan_gen, gan_disc, optimizer_gan_g)

            for i in range(5):
                total_loss_inv += train_inv((x, x_mask, y),
                                            Seq2Seq_model_bert,
                                            gan_gen,
                                            inverter,
                                            optimizer_inv,
                                            criterion_ce,
                                            criterion_js,
                                            criterion_mse,
                                            gamma=0.5)
            if niter % 100 == 0:
                # decaying noise
                Seq2Seq_model_bert.noise_std *= 0.995
                logging(
                    f'epoch {epoch}, niter {niter}:Loss_Seq2Seq: {total_loss_Seq2Seq / niter}, Loss_gan_g: {total_loss_gan_g / niter}, Loss_gan_d: {total_loss_gan_d / niter / 5}, Loss_inv: {total_loss_inv / niter / 5}'
                )

                if niter % 30000 == 0:
                    noise = torch.ones(Config.batch_size,
                                       Config.super_hidden_size)
                    noise.normal_(0, 1)
                    evaluate_generator(noise, Seq2Seq_model_bert, gan_gen)
