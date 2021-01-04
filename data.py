from torch.utils.data import Dataset
from tools import read_standard_data4Test, logging, read_standard_data
import torch
from config import Config
from transformers import BertTokenizer


class Baseline_Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.datas, self.labels = read_standard_data(self.path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data_tokens = []
        self.data_idx = []
        self.label_idx = []
        self.data_mask = []
        self.data2tokens()
        self.token2idx()

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.datas:
            tokens = self.tokenizer.tokenize(sen)[:Config.sen_len - 2]
            self.data_tokens.append(['[CLS]'] + tokens + ['[SEP]'])

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.data_mask.append([1] * len(tokens))

        sen_len = Config.sen_len
        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < sen_len:
                self.data_idx[i] += [0] * (sen_len - len(self.data_idx[i]))
                self.data_mask[i] += [0] * (sen_len - len(self.data_mask[i]))

        for label in self.labels:
            self.label_idx.append(label)

    def __getitem__(self, item):
        return torch.tensor(self.data_idx[item]), torch.tensor(
            self.data_mask[item]), torch.tensor(self.label_idx[item])

    def __len__(self):
        return len(self.datas)


class Seq2Seq_DataSet(Dataset):
    """x = seq, y = seq + 1

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, path):
        super(Seq2Seq_DataSet, self).__init__()
        self.path = path
        self.datas, self.labels = read_standard_data(self.path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data_tokens = []
        self.label_tokens = []
        self.data_idx = []
        self.label_idx = []
        self.data_mask = []
        self.classification_label = []
        self.data2tokens()
        self.token2idx()
        self.transfor()

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.datas:
            data_tokens = self.tokenizer.tokenize(sen)[:Config.sen_len - 1]
            data_tokens.append('[SEP]')
            self.data_tokens.append(data_tokens)
            label_tokens = ['[CLS]']
            label_tokens += self.tokenizer.tokenize(sen)[:Config.sen_len - 1]
            self.label_tokens.append(label_tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.data_mask.append([1] * len(tokens))

        for tokens in self.label_tokens:
            self.label_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))

        sen_len = Config.sen_len
        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < sen_len:
                self.data_idx[i] += [0] * (sen_len - len(self.data_idx[i]))
                self.label_idx[i] += [0] * (sen_len - len(self.label_idx[i]))
                self.data_mask[i] += [0] * (sen_len - len(self.data_mask[i]))

        for label in self.labels:
            self.classification_label.append(label)

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.data_mask = torch.tensor(self.data_mask)
        self.label_idx = torch.tensor(self.label_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.data_mask[item], self.label_idx[
            item], self.classification_label[item]

    def __len__(self):
        return len(self.datas)


class Seq2Seq_DataSet4Test(Dataset):
    """x = seq, y = seq + 1

    Args:
        Dataset ([type]): [description]
    """
    def __init__(self, path):
        super(Seq2Seq_DataSet4Test, self).__init__()
        self.path = path
        self.datas, self.labels = read_standard_data4Test(self.path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data_tokens = []
        self.label_tokens = []
        self.data_idx = []
        self.label_idx = []
        self.data_mask = []
        self.classification_label = []
        self.data2tokens()
        self.token2idx()
        self.transfor()

    def data2tokens(self):
        logging(f'{self.path} in data2tokens')
        for sen in self.datas:
            data_tokens = self.tokenizer.tokenize(sen)[:Config.sen_len - 1]
            data_tokens.append('[SEP]')
            self.data_tokens.append(data_tokens)
            label_tokens = ['[CLS]']
            label_tokens += self.tokenizer.tokenize(sen)[:Config.sen_len - 1]
            self.label_tokens.append(label_tokens)

    def token2idx(self):
        logging(f'{self.path} in token2idx')
        for tokens in self.data_tokens:
            self.data_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))
            self.data_mask.append([1] * len(tokens))

        for tokens in self.label_tokens:
            self.label_idx.append(self.tokenizer.convert_tokens_to_ids(tokens))

        sen_len = Config.sen_len
        for i in range(len(self.data_idx)):
            if len(self.data_idx[i]) < sen_len:
                self.data_idx[i] += [0] * (sen_len - len(self.data_idx[i]))
                self.label_idx[i] += [0] * (sen_len - len(self.label_idx[i]))
                self.data_mask[i] += [0] * (sen_len - len(self.data_mask[i]))

        for label in self.labels:
            self.classification_label.append(label)

    def transfor(self):
        self.data_idx = torch.tensor(self.data_idx)
        self.data_mask = torch.tensor(self.data_mask)
        self.label_idx = torch.tensor(self.label_idx)
        self.classification_label = torch.tensor(self.classification_label)

    def __getitem__(self, item):
        return self.data_idx[item], self.data_mask[item], self.label_idx[
            item], self.classification_label[item]

    def __len__(self):
        return len(self.datas)
