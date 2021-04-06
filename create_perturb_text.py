from transformers import BertTokenizer
from model import Seq2Seq_bert, LSTM_G, LSTM_A
from baseline_module.baseline_model_builder import BaselineModelBuilder
from data import AGNEWS_Dataset, IMDB_Dataset, SNLI_Dataset, SST2_Dataset
from config import AttackConfig
from torch.utils.data import DataLoader
from calc_BertScore_ppl import calc_bert_score_ppl
from tools import logging
import torch
import os
import numpy as np
import time


def perturb(data, Seq2Seq_model, gan_gen, gan_adv, baseline_model, cands_dir,
            refs_dir, attack_vocab, search_bound, samples_num):
    # Turn on evaluation mode which disables dropout.
    Seq2Seq_model.eval()
    gan_gen.eval()
    gan_adv.eval()
    baseline_model.eval()
    with torch.no_grad():
        attack_num = 0
        attack_success_num = 0
        gen_time = 0
        test_time = 0
        total_time = time.time()
        val_time = 0
        search_time = 0
        output_time = 0
        with open(cands_dir, "w") as f, open(refs_dir, "w") as f_1:
            for x, x_mask, y, label in data:
                x, x_mask, y, label = x.to(train_device), x_mask.to(
                    train_device), y.to(train_device), label.to(train_device)

                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                gen_time -= time.time()
                # c: [batch, sen_len, hidden_size]
                c = Seq2Seq_model(x, x_mask, is_noise=False, encode_only=True)
                # z: [batch, seq_len, super_hidden_size]
                z = gan_adv(c)
                gen_time += time.time()

                val_time -= time.time()

                if baseline_model_name == 'Bert':
                    x_type = torch.zeros(y.shape,
                                         dtype=torch.int64).to(train_device)
                    skiped = label != baseline_model(y, x_type,
                                                     x_mask).argmax(dim=1)
                else:
                    skiped = label != baseline_model(y).argmax(dim=1)

                val_time += time.time()

                y = y.to(torch.device('cpu'))

                for i in range(len(y)):
                    if skiped[i].item():
                        continue

                    attack_num += 1

                    search_time -= time.time()

                    perturb_x, successed_mask, g_t, t_t = search_fast(
                        Seq2Seq_model,
                        gan_gen,
                        baseline_model,
                        label[i],
                        z[i],
                        samples_num=samples_num,
                        search_bound=search_bound)

                    search_time += time.time()

                    gen_time += g_t
                    test_time += t_t

                    output_time -= time.time()

                    if attack_vocab:
                        for stop in range(len(y[i]), 0, -1):
                            if y[i][stop - 1].item() != 0:
                                if y[i][stop - 1].item() != 2:
                                    break
                        f_1.write(' '.join([
                            attack_vocab.get_word(token)
                            for token in y[i][:stop]
                        ]) + "\n")

                        for n, perturb_x_sample in enumerate(perturb_x):
                            if successed_mask[n].item():
                                for stop in range(len(perturb_x_sample), 0,
                                                  -1):
                                    if perturb_x_sample[stop - 1].item() != 0:
                                        if perturb_x_sample[stop -
                                                            1].item() != 2:
                                            break
                                f.write(' '.join([
                                    attack_vocab.get_word(token)
                                    for token in perturb_x_sample[:stop]
                                ]))
                                f.write('\n')
                                attack_success_num += 1
                                break
                        else:
                            for stop in range(len(perturb_x[0]), 0, -1):
                                if perturb_x[0][stop - 1].item() != 0:
                                    if perturb_x[0][stop - 1].item() != 2:
                                        break
                            f.write(' '.join([
                                attack_vocab.get_word(token)
                                for token in perturb_x[0][:stop]
                            ]))
                            f.write('\n')

                    else:
                        for stop in range(len(y[i]), 0, -1):
                            if y[i][stop - 1].item() != 0:
                                if y[i][stop - 1].item() != 102:
                                    break
                        f_1.write(' '.join(
                            tokenizer.convert_ids_to_tokens(y[i][:stop])) +
                                  "\n")

                        for n, perturb_x_sample in enumerate(perturb_x):
                            if successed_mask[n].item():
                                for stop in range(len(perturb_x_sample), 0,
                                                  -1):
                                    if perturb_x_sample[stop - 1].item() != 0:
                                        if perturb_x_sample[stop -
                                                            1].item() != 102:
                                            break
                                f.write(' '.join(
                                    tokenizer.convert_ids_to_tokens(
                                        perturb_x_sample[:stop])))
                                f.write('\n')
                                attack_success_num += 1
                                break
                        else:
                            for stop in range(len(perturb_x[0]), 0, -1):
                                if perturb_x[0][stop - 1].item() != 0:
                                    if perturb_x[0][stop - 1].item() != 102:
                                        break
                            f.write(' '.join(
                                tokenizer.convert_ids_to_tokens(
                                    perturb_x[0][:stop])))
                            f.write('\n')

                    output_time += time.time()

        total_time = time.time() - total_time

    return attack_success_num / attack_num, gen_time, test_time, total_time, val_time, search_time, output_time


def search_fast(Seq2Seq_model, generator, baseline_model, label, z,
                samples_num, search_bound):
    # z: [sen_len, super_hidden_size]
    Seq2Seq_model.eval()
    generator.eval()
    baseline_model.eval()
    with torch.no_grad():
        g_t = time.time()
        dist = []
        if samples_num > 100:
            for i in range(int(samples_num / 100)):
                # search_z: [samples_num, sen_len, super_hidden_size]
                search_z = z.repeat(100, 1, 1)
                delta = torch.FloatTensor(search_z.size()).uniform_(
                    -1 * search_bound, search_bound)
                dist = dist + [np.sqrt(np.sum(x**2)) for x in delta.numpy()]
                delta = delta.to(train_device)
                search_z += delta
                # pertub_hidden: [samples_num, sen_len, hidden_size]
                perturb_hidden = generator(search_z)
                # pertub_x: [samples_num, seq_len]
                if i == 0:
                    perturb_x = Seq2Seq_model.decode(perturb_hidden).argmax(
                        dim=2)
                else:
                    perturb_x = torch.cat(
                        (perturb_x,
                         Seq2Seq_model.decode(perturb_hidden).argmax(dim=2)),
                        dim=0)
            else:
                dist = np.array(dist)
        else:
            # search_z: [samples_num, sen_len, super_hidden_size]
            search_z = z.repeat(samples_num, 1, 1)
            delta = torch.FloatTensor(search_z.size()).uniform_(
                -1 * search_bound, search_bound)
            dist = np.array([np.sqrt(np.sum(x**2)) for x in delta.numpy()])
            delta = delta.to(train_device)
            search_z += delta
            # pertub_hidden: [samples_num, sen_len, hidden_size]
            perturb_hidden = generator(search_z)
            # pertub_x: [samples_num, seq_len]
            perturb_x = Seq2Seq_model.decode(perturb_hidden).argmax(dim=2)

        g_t = time.time() - g_t
        t_t = time.time()

        if baseline_model_name == 'Bert':
            perturb_x_mask = torch.ones(perturb_x.shape, dtype=torch.int64)
            # mask before [SEP]
            for i in range(perturb_x.shape[0]):
                for word_idx in range(perturb_x.shape[1]):
                    if perturb_x[i][word_idx].item() == 102:
                        perturb_x_mask[i][word_idx + 1:] = 0
                        break
            perturb_x_mask = perturb_x_mask.to(train_device)
            perturb_x_type = torch.zeros(perturb_x.shape,
                                         dtype=torch.int64).to(train_device)
            # perturb_label: [samples_num]
            perturb_label = baseline_model(perturb_x, perturb_x_type,
                                           perturb_x_mask).argmax(dim=1)
            t_t = time.time() - t_t
        else:
            perturb_label = baseline_model(perturb_x).argmax(dim=1)
            t_t = time.time() - t_t

        successed_mask = perturb_label != label

        sorted_perturb_x = []
        sorted_successed_mask = []
        for _, x, y in sorted(zip(dist, perturb_x, successed_mask),
                              key=lambda pair: pair[0]):
            sorted_perturb_x.append(x)
            sorted_successed_mask.append(y)

    return sorted_perturb_x, sorted_successed_mask, g_t, t_t


def build_dataset(attack_vocab, batch_size):
    if dataset == 'SNLI':
        train_dataset_orig = SNLI_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=True)
        test_dataset_orig = SNLI_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=True)
    elif dataset == 'AGNEWS':
        train_dataset_orig = AGNEWS_Dataset(
            train_data=True,
            attack_vocab=attack_vocab,
            debug_mode=True,
        )
        test_dataset_orig = AGNEWS_Dataset(train_data=False,
                                           attack_vocab=attack_vocab,
                                           debug_mode=True)
    elif dataset == 'IMDB':
        train_dataset_orig = IMDB_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=True)
        test_dataset_orig = IMDB_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=True)
    elif dataset == 'SST2':
        train_dataset_orig = SST2_Dataset(train_data=True,
                                          attack_vocab=attack_vocab,
                                          debug_mode=True)
        test_dataset_orig = SST2_Dataset(train_data=False,
                                         attack_vocab=attack_vocab,
                                         debug_mode=True)
    train_data = DataLoader(train_dataset_orig,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=batch_size,
                           shuffle=False,
                           num_workers=4)
    return train_data, test_data


if __name__ == '__main__':
    train_device = torch.device('cuda:1')
    dataset = 'SST2'
    baseline_model_name = 'LSTM'
    search_bound = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    samples_num = [20, 100, 1000, 2000, 5000]

    cur_dir = './output/gan_model/SST2/LSTM/1616592725/models/epoch29/'  # gan_adv gan_gen Seq2Seq_model
    output_dir = f'./texts/OUR/{dataset}/{baseline_model_name}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    baseline_model_builder = BaselineModelBuilder(dataset,
                                                  baseline_model_name,
                                                  train_device,
                                                  is_load=True)

    Seq2Seq_model = Seq2Seq_bert(
        baseline_model_builder.vocab.num if baseline_model_builder.
        vocab else AttackConfig.vocab_size).to(train_device)
    Seq2Seq_model.load_state_dict(
        torch.load(cur_dir + 'Seq2Seq_model.pt', map_location=train_device))

    gan_gen = LSTM_G(AttackConfig.super_hidden_size,
                     AttackConfig.hidden_size,
                     num_layers=3).to(train_device)
    gan_gen.load_state_dict(
        torch.load(cur_dir + 'gan_gen.pt', map_location=train_device))

    gan_adv = LSTM_A(AttackConfig.hidden_size,
                     AttackConfig.super_hidden_size,
                     num_layers=3).to(train_device)
    gan_adv.load_state_dict(
        torch.load(cur_dir + 'gan_adv.pt', map_location=train_device))

    _, test_data = build_dataset(baseline_model_builder.vocab, batch_size=128)

    for samples in samples_num:
        for bound in search_bound:
            refs_dir = output_dir + f'/refs_{bound}_{samples}.txt'
            cands_dir = output_dir + f'/cands_{bound}_{samples}.txt'
            attack_acc, gen_time, test_time, total_time, val_time, search_time, output_time = perturb(
                test_data, Seq2Seq_model, gan_gen, gan_adv,
                baseline_model_builder.net, cands_dir, refs_dir,
                baseline_model_builder.vocab, bound, samples)
            logging(f'search_bound={bound}, sample={samples}')
            logging(f'attack_acc={attack_acc}')

            ppl, bert_score = calc_bert_score_ppl(cands_dir, refs_dir,
                                                  train_device)
            logging(f'ppl={ppl}')
            logging(f'bert_score={bert_score}')
            logging(f'gen_time={gen_time}')
            logging(f'test_time={test_time}')
            logging(f'total_time={total_time}')
            logging(f'val_time={val_time}')
            logging(f'search_time={search_time}')
            logging(f'output_time={output_time}')
