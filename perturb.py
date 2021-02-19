from config import AttackConfig
from transformers import BertTokenizer
import torch


def perturb(data, Seq2Seq_model, gan_gen, gan_adv, baseline_model, dir,
            attack_vocab):
    # Turn on evaluation mode which disables dropout.
    Seq2Seq_model.eval()
    gan_gen.eval()
    gan_adv.eval()
    baseline_model.eval()
    with torch.no_grad():
        attack_succeeded_num = 0
        attack_num = 0
        attack_count_num = 0
        attack_succeeded_idx_num = [0] * AttackConfig.perturb_sample_num
        with open(dir, "a") as f:
            for x, x_mask, y, label in data:
                x, x_mask, y, label = x.to(
                    AttackConfig.train_device), x_mask.to(
                        AttackConfig.train_device), y.to(
                            AttackConfig.train_device), label.to(
                                AttackConfig.train_device)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                # c: [batch, sen_len, hidden_size]
                c = Seq2Seq_model(x, x_mask, is_noise=False, encode_only=True)
                # z: [batch, seq_len, super_hidden_size]
                z = gan_adv(c)

                for i in range(len(y)):

                    perturb_x, successed_mask, counter, successed = search_fast(
                        Seq2Seq_model,
                        gan_gen,
                        baseline_model,
                        label[i],
                        z[i],
                        samples_num=AttackConfig.perturb_sample_num,
                        right=AttackConfig.perturb_search_bound)
                    attack_count_num += counter
                    attack_num += 1

                    if successed:
                        attack_succeeded_num += 1
                        for index in range(len(successed_mask)):
                            if successed_mask[index].item():
                                attack_succeeded_idx_num[index] += 1

                    if i % 100 == 0:
                        if attack_vocab:
                            f.write(
                                "==================================================\n"
                            )
                            f.write('Orginal sentence: \n')
                            f.write(' '.join([
                                attack_vocab.get_word(token) for token in y[i]
                            ]) + "\n" * 2)

                            if successed:
                                f.write(
                                    '==============Attark succeed!=================\n'
                                )
                                f.write(f'5 samples try for {counter} times\n')
                                f.write('Attack successed sample: \n')
                                for index in range(len(successed_mask)):
                                    if successed_mask[index].item():
                                        f.write(' '.join([
                                            attack_vocab.get_word(token)
                                            for token in perturb_x[index]
                                        ]))
                                        f.write('\n')

                                f.write('\nAll attack samples as follows: \n')
                                for i, perturb_x_sample in enumerate(
                                        perturb_x):
                                    f.write(' '.join([
                                        attack_vocab.get_word(token)
                                        for token in perturb_x_sample
                                    ]))
                                    if successed_mask[i]:
                                        f.write('    attact successed!')
                                    else:
                                        f.write('    attact failed!')
                                    f.write('\n')
                                f.write(
                                    '\n============================================\n'
                                )
                                f.flush()
                            else:
                                f.write(
                                    '==============Attack Failed ==================\n'
                                )
                                f.write(f'5 samples try for {counter} times\n')
                                f.write('\nAll attack samples as follows: \n')
                                for i, perturb_x_sample in enumerate(
                                        perturb_x):
                                    f.write(' '.join([
                                        attack_vocab.get_word(token)
                                        for token in perturb_x_sample
                                    ]))
                                    f.write('\n')
                        else:
                            f.write(
                                "==================================================\n"
                            )
                            f.write('Orginal sentence: \n')
                            f.write(' '.join(
                                tokenizer.convert_ids_to_tokens(y[i])) +
                                    "\n" * 2)

                            if successed:
                                f.write(
                                    '==============Attark succeed!=================\n'
                                )
                                f.write(f'5 samples try for {counter} times\n')
                                f.write('Attack successed sample: \n')
                                for index in range(len(successed_mask)):
                                    if successed_mask[index].item():
                                        f.write(' '.join(
                                            tokenizer.convert_ids_to_tokens(
                                                perturb_x[index])))
                                        f.write('\n')

                                f.write('\nAll attack samples as follows: \n')
                                for i, perturb_x_sample in enumerate(
                                        perturb_x):
                                    f.write(' '.join(
                                        tokenizer.convert_ids_to_tokens(
                                            perturb_x_sample)))
                                    if successed_mask[i]:
                                        f.write('    attact successed!')
                                    else:
                                        f.write('    attact failed!')
                                    f.write('\n')
                                f.write(
                                    '\n============================================\n'
                                )
                                f.flush()
                            else:
                                f.write(
                                    '==============Attack Failed ==================\n'
                                )
                                f.write(f'5 samples try for {counter} times\n')
                                f.write('\nAll attack samples as follows: \n')
                                for i, perturb_x_sample in enumerate(
                                        perturb_x):
                                    f.write(' '.join(
                                        tokenizer.convert_ids_to_tokens(
                                            perturb_x_sample)))
                                    f.write('\n')

            f.write(
                f'attact success acc:{attack_succeeded_num / attack_num}\n\n')

            for i in range(AttackConfig.perturb_sample_num):
                f.write(
                    f'sample {i} attact success acc:{attack_succeeded_idx_num[i] / attack_num}\n\n'
                )
            f.write(f'avg attack try times:{attack_count_num / attack_num}\n')
    return attack_succeeded_num / attack_num, [
        x / attack_num for x in attack_succeeded_idx_num
    ], attack_count_num / attack_num


def search_fast(Seq2Seq_model, generator, baseline_model, label, z,
                samples_num, right):
    """search for adversary sample

    Args:
        Seq2Seq_model (nn.Module): input: [batch, seq_len], output: [batch, seq_len]
        generator (nn.Module): input: super_hidden_size , output: hidden_size
        baseline_model (nn.Module): input: [batch, seq_len], output: [batch, 4]
        label (torch.tensor): [1]
        z (torch.tensor): [sen_len, super_hidden_size]
        samples_num (int, optional): num of samples. Defaults to 5.
        right (float, optional): begining search boundary. Defaults to 0.005.

    Returns:
        [type]: [description]
    """
    # z: [sen_len, super_hidden_size]
    Seq2Seq_model.eval()
    generator.eval()
    baseline_model.eval()
    with torch.no_grad():
        search_bound = right
        counter = 0
        while counter < AttackConfig.perturb_search_times:
            # search_z: [samples_num, sen_len, super_hidden_size]
            search_z = z.repeat(samples_num, 1, 1)
            delta = torch.FloatTensor(search_z.size())  # 1.122
            delta[0] = 0.0
            for i in range(1, samples_num):
                delta[i].uniform_(-1 * search_bound * (2**i),
                                  search_bound * (2**i))
            delta = delta.to(AttackConfig.train_device)
            search_z += delta
            # pertub_hidden: [samples_num, sen_len, hidden_size]
            perturb_hidden = generator(search_z)
            # pertub_x: [samples_num, seq_len]
            perturb_x = Seq2Seq_model.decode(perturb_hidden).argmax(dim=2)
            if AttackConfig.baseline_model == 'Bert':
                perturb_x_mask = torch.ones(perturb_x.shape)
                # mask before [SEP]
                for i in range(perturb_x.shape[0]):
                    for word_idx in range(perturb_x.shape[1]):
                        if perturb_x[i][word_idx].item() == 102:
                            perturb_x_mask[i][word_idx + 1:] = 0
                            break
                perturb_x_mask = perturb_x_mask.to(AttackConfig.train_device)
                # perturb_label: [samples_num]
                perturb_label = baseline_model(perturb_x,
                                               perturb_x_mask).argmax(dim=1)
            else:
                perturb_label = baseline_model(perturb_x).argmax(dim=1)

            successed = False
            successed_mask = perturb_label != label
            for i in range(successed_mask.shape[0]):
                if successed_mask[i].item():
                    successed = True
                    break

            if successed:
                counter += 1
                return perturb_x, successed_mask, counter, successed
            else:
                counter += 1
                search_bound *= 2

    return perturb_x, successed_mask, counter, successed
