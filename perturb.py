from config import Config
from transformers import BertTokenizer
import torch


def perturb(data, Seq2Seq_model, gan_gen, inverter, baseline_model, dir):
    # Turn on evaluation mode which disables dropout.
    Seq2Seq_model.eval()
    gan_gen.eval()
    inverter.eval()
    baseline_model.eval()

    with open(dir, "a") as f:
        for x, x_mask, y, label in data:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # c: [batch, sen_len, hidden_size]
            c = Seq2Seq_model(x, x, x_mask, is_noise=False, encode_only=True)
            # z: [batch, seq_len, super_hidden_size]
            z = inverter(c).data

            for i in range(Config.batch_size):
                f.write("==================================================\n")
                f.write('Orginal sentence: \n')
                f.write(' '.join(tokenizer.convert_ids_to_tokens(x[i])) +
                        "\n" * 2)

                perturb_x, perturb_x_mask, perturb_label, successed_mask, bound_distence, counter, successed = search_fast(
                    Seq2Seq_model,
                    gan_gen,
                    baseline_model,
                    label[i],
                    z[i],
                    samples_num=20)

                if successed:
                    f.write('==============Attark succeed!=================\n')
                    f.write(f'20 samples try for {counter} times\n')
                    f.write('Attack successed sample: \n')
                    for i, successed_x in enumerate(
                            perturb_x.masked_select(successed_mask)):
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(successed_x)))
                        f.write(str(bound_distence[i]))
                        f.write('\n')

                    f.write('\nAll attack samples as follows: \n')
                    for i, perturb_x_sample in enumerate(perturb_x):
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(perturb_x_sample)))
                        f.write(str(bound_distence[i]))
                        f.write('\n')
                    f.write('\n============================================\n')
                    f.flush()
                else:
                    f.write('==============Attack Failed ==================\n')
                    f.write(f'20 samples try for {counter} times\n')
                    f.write('\nAll attack samples as follows: \n')
                    for i, perturb_x_sample in enumerate(perturb_x):
                        f.write(' '.join(
                            tokenizer.convert_ids_to_tokens(perturb_x_sample)))
                        f.write(str(bound_distence[i]))
                        f.write('\n')


def search_fast(Seq2Seq_model,
                generator,
                baseline_model,
                label,
                z,
                samples_num=20,
                right=0.005):
    """search for adversary sample

    Args:
        Seq2Seq_model (nn.Module): input: [batch, seq_len], output: [batch, seq_len]
        generator (nn.Module): input: super_hidden_size , output: hidden_size
        baseline_model (nn.Module): input: [batch, seq_len], output: [batch, 4]
        label (torch.tensor): [1]
        z (torch.tensor): [sen_len, super_hidden_size]
        samples_num (int, optional): num of samples. Defaults to 20.
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
        while counter < 5:
            # search_z: [samples_num, sen_len, super_hidden_size]
            search_z = z.repeat(samples_num, z.shape[0], z.shape[1])
            delta = torch.FloatTensor(search_z.size()).uniform_(
                -1 * search_bound, search_bound)
            # bound_distence: [samples_num, sen_len, super_hidden_size]
            bound_distence = torch.abs(delta)
            search_z += delta
            # pertub_hidden: [samples_num, sen_len, hidden_size]
            perturb_hidden = generator(search_z)
            # pertub_x: [samples_num, seq_len]
            perturb_x = Seq2Seq_model.decode(perturb_hidden).argmax(dim=2)
            perturb_x_mask = torch.ones(perturb_x.shape)
            # mask before [SEP]
            for i in range(perturb_x.shape[0]):
                for word_idx in range(perturb_x.shape[1]):
                    if perturb_x[i][word_idx].data == 102:
                        perturb_x_mask[i][word_idx + 1:] = 0
                        break
            # perturb_label: [samples_num]
            perturb_label = baseline_model(perturb_x,
                                           perturb_x_mask).argmax(dim=1)

            successed = False
            successed_mask = perturb_label != label
            for i in range(successed_mask.shape[0]):
                if successed_mask[i].data is True:
                    successed = True
                    break

            if successed:
                return perturb_x, perturb_x_mask, perturb_label, successed_mask, bound_distence.sum(
                    dim=2), counter, successed
            else:
                counter += 1
                search_bound *= 2

    return perturb_x, perturb_x_mask, perturb_label, successed_mask, bound_distence.sum(
        dim=2), counter, successed