import argparse
import copy
import os
import re

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline_config import *
from baseline_data import baseline_MyDataset
from baseline_model_builder import BaselineModelBuilder
from baseline_tools import parse_bool, logging, get_time, save_pkl_obj, load_pkl_obj
from baseline_vocab import baseline_Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=baseline_config_dataset.keys(), default='IMDB')
parser.add_argument('--model', choices=baseline_config_models_list, default='Bert')
parser.add_argument('--save_acc_limit', help='set a acc lower limit for saving model',
                    type=float, default=0.85)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--load_model', choices=[True, False], default='no', type=parse_bool)
parser.add_argument('--cuda', type=str, default='3')
parser.add_argument('--skip_loss', type=float, default=0.15)
args = parser.parse_args()


dataset_name = args.dataset
dataset_config = baseline_config_dataset[dataset_name]
batch = args.batch
lr = args.lr
note = args.note
is_load_model = args.load_model
model_name = args.model
using_bert = model_name == 'Bert'
device = torch.device('cuda:'+args.cuda)


# prepare dataset
temp_path = f'./baseline_models/traindata_{dataset_name}_{model_name}.pkl'
if os.path.exists(temp_path) and False:
    train_dataset = load_pkl_obj(temp_path)
else:
    train_dataset = baseline_MyDataset(dataset_name, dataset_config.train_data_path, using_bert)
    if model_name == 'Bert':
        vocab = None
    else:
        vocab = baseline_Vocab(train_dataset.data_token, is_using_pretrained=True,
                               vocab_limit_size=dataset_config.vocab_limit_size,
                               word_vec_file_path=dataset_config.pretrained_word_vectors_path)
    train_dataset.token2seq(vocab, dataset_config.padding_maxlen)
    save_pkl_obj(train_dataset, temp_path)

vocab = train_dataset.vocab

temp_path = f'./baseline_models/testdata_{dataset_name}_{model_name}.pkl'
if os.path.exists(temp_path):
    test_dataset = load_pkl_obj(temp_path)
else:
    test_dataset = baseline_MyDataset(dataset_name, dataset_config.test_data_path, using_bert)
    test_dataset.token2seq(vocab, dataset_config.padding_maxlen)
    save_pkl_obj(test_dataset, temp_path)

# ----------------------------------------------
model = BaselineModelBuilder(dataset_name, model_name, device, is_load_model, vocab=train_dataset.vocab)

train_dataset = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=batch)

if is_load_model: logging(f'loading model from {dataset_name} {model_name} {baseline_config_model_load_path[dataset_name][model_name]}')
else: logging(f'training model {dataset_name} {model_name} from scratch')
optimizer = optim.AdamW(
    [
        {'params': model.net.bert_model.parameters(), 'lr': 1e-5},
        {'params': model.net.fc.parameters(), 'lr': lr}
    ], lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1
)
criterion = nn.CrossEntropyLoss().to(device)



def train():
    loss_mean = 0.0
    model.set_training_mode()
    with tqdm(total=len(train_dataset), desc='train') as pbar:
        for temp in train_dataset:
            if using_bert:
                x, types, masks = temp[0].to(device), temp[1].to(device), temp[2].to(device)
            else:
                x = temp[0].to(device)
                types = masks = None
            y = temp[3].to(device)

            logits = model(x, types, masks)
            loss = criterion(logits, y)
            loss_mean += loss.item()
            if loss.item() > args.skip_loss:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pbar.set_postfix_str(f'loss {loss.item():.5f}')
            pbar.update(1)

    return loss_mean / len(train_dataset)

@torch.no_grad()
def evaluate():
    model.set_eval_mode()
    loss_mean = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(test_dataset), desc='evaluate') as pbar:
        for temp in train_dataset:
            if using_bert:
                x, types, masks = temp[0].to(device), temp[1].to(device), temp[2].to(device)
            else:
                x = temp[0].to(device)
                types = masks = None
            y = temp[3].to(device)

            logits = model(x, types, masks)
            loss = criterion(logits, y)
            loss_mean += loss.item()

            predicts = logits.argmax(dim=-1)
            correct += predicts.eq(y).float().sum().item()
            total += x.size()[0]

            pbar.set_postfix_str(f'loss {loss.item():.5f} acc {correct/total:.5f}')
            pbar.update(1)

    return loss_mean / len(test_dataset), correct / total


def main():
    best_path = baseline_config_model_load_path[dataset_name].get(model_name)
    best_state = None
    best_acc = 0.0 if is_load_model == False else float(re.findall("_\d.\d+_", best_path)[0][1:-1])
    save_acc_limit = args.save_acc_limit
    epoch = args.epoch

    for ep in range(epoch):
        logging(f'epoch {ep} start train')
        train_loss = train()
        logging(f'epoch {ep} start evaluate')
        evaluate_loss, acc = evaluate()
        if acc > best_acc:
            best_acc = acc
            best_path = baseline_config_model_save_path_format.format(dataset_name, model_name,
                                                                      acc, get_time(), note)
            best_state = copy.deepcopy(model.net.state_dict())

        if (ep+1) % (epoch // 3) == 0 and best_acc > save_acc_limit and best_state != None:
            logging(f'saving best model acc {best_acc:.5f} in {best_path}')
            torch.save(best_state, best_path)
            best_state = None


        logging(f'epoch {ep} done! train_loss {train_loss} evaluate_loss {evaluate_loss}'
                f'acc {acc} now best_acc {best_acc}')


if __name__ == '__main__':
    main()