import os
import time
from model import MLP_D, MLP_G, MLP_I, Seq2SeqAE, JSDistance
from data import MyDataset
from tools import logging
from config import Config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from vocab import Vocab
import torch

print('Using cuda device gpu: ' + Config.config_device.type)

Config.outf = str(int(time.time()))
cur_dir = Config.outputdir + '/' + Config.outf
# make output directory if it doesn't already exist
if not os.path.isdir(Config.output_dir):
    os.makedirs(Config.output_dir)
if not os.path.isdir(cur_dir):
    os.makedirs(cur_dir)
    os.makedirs(cur_dir + '/models')
print('Saving into directory' + cur_dir)

if Config.load_pretrain:
    print("Loading pretrained models from " + cur_dir)
else:
    print("Creating new experiment at " + cur_dir)

train_dataset_orig = MyDataset(Config.train_data_path)
test_dataset_orig = MyDataset(Config.test_data_path)
vocab = Vocab(origin_data_tokens=train_dataset_orig.data_tokens,
              is_using_pretrained=False,
              vocab_limit_size=Config.vocab_limit_size)
train_dataset_orig.token2idx(vocab, Config.maxlen)
test_dataset_orig.token2idx(vocab, Config.maxlen)
train_data = DataLoader(train_dataset_orig,
                        batch_size=Config.batch_size,
                        shuffle=False)
test_data = DataLoader(test_dataset_orig,
                       batch_size=Config.batch_size,
                       shuffle=False)

gan_gen = MLP_G(300, 300).to(Config.train_device)
gan_disc = MLP_D(300, 1).to(Config.train_device)
inverter = MLP_I(300, 100).to(Config.train_device)
autoencoder = Seq2SeqAE(len(vocab), Config.maxlen, 100, 300).to(Config.train_device)

print(autoencoder)
print(inverter)
print(gan_gen)
print(gan_disc)

optimizer_ae = optim.Adam(autoencoder.parameters(), lr=3e-4)
optimizer_inv = optim.Adam(inverter.parameters(), lr=1e-5, betas=(0.9, 0.999))
optimizer_gan_g = optim.Adam(gan_gen.parameters(), lr=5e-5, betas=(0.9, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=1e-5,
                             betas=(0.9, 0.999))

criterion_ce = nn.CrossEntropyLoss().to(Config.train_device)
criterion_mse = nn.MSELoss().to(Config.train_device)
criterion_js = JSDistance().to(Config.train_device)

print('Training...')


def train():
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(Config.epochs):
        logging(f'epoch {epoch} start')
        logging(f'epoch {epoch} train autoencoder')

        loss_mean = 0.0

        for x, y in tqdm(train_data):
            x, y = x.to(Config.train_device), y.to(Config.train_device)
            logits = autoencoder(x, is_noise=True)
            logits = logits.reshape(x.size()[0] * x.size()[1], -1)
            optimizer_ae.zero_grad()
            loss = criterion_ce(logits, x.view(-1))
            loss_mean += loss.item()
            loss.backward()
            optimizer_ae.step()
        loss_mean /= len(train_data)
        print(f"epoch {epoch} loss is {loss_mean}")
        if loss_mean < 0.1:
            break

    with torch.no_grad():
        autoencoder.eval()
        batch_generated = []
        for x, y in test_data:
            x, y = x.to(Config.train_device), y.to(Config.train_device)
            logits = autoencoder(x, is_noise=False)
            res = torch.argmax(logits, dim=-1)
            generated = [[0 for _ in range(Config.maxlen)] for _ in range(Config.batch_size)]
            for i, sample in enumerate(res):
                for j, idx in enumerate(sample):
                    generated[i][j] = vocab.get_word(idx.item())
                generated[i] = ' '.join(generated[i])
            batch_generated += generated

        for i in range(len(test_data)):
            logging(str(i))
            print(test_dataset_orig.datas[i])
            print()
            print(batch_generated[i])


def train_ae(x, y, total_loss_ae, autoencoder, optimizer_ae, criterion_ce):
    autoencoder.train()
    optimizer_ae.zero_grad()

    flattened_y = y.view(-1)
    mask = flattened_y.gt(0)
    masked_y = flattened_y.masked_select(mask)
    logits_mask = mask.unsqueeze(1).expand(mask.size(0), Config.vocab_limit_size)

    logits = autoencoder(x, is_noise=True)

    flattened_logits = logits.view(-1, Config.vocab_limit_size)

    masked_logits = flattened_logits.masked_select(logits_mask).view(-1, Config.vocab_limit_size)
    loss = criterion_ce(masked_logits, masked_y)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1)
    optimizer_ae.step()

    total_loss_ae += loss.data


if __name__ == '__main__':
    train()
