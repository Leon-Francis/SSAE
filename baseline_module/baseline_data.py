import random
import re

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from baseline_config import baseline_config_dataset, baseline_BertConfig
from baseline_tools import logging, read_standard_data


class baseline_Tokenizer():
    def __init__(self):
        pass

    def pre_process(self, text: str)->str:
        text = text.lower().strip()
        text = re.sub(r"<br />", "", text)
        text = re.sub(r'(\W)(?=\1)', '', text)
        text = re.sub(r"([.!?,])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
        return text.strip()

    def normal_token(self, text: str)->[str]:
        return [tok for tok in text.split() if not tok.isspace()]


    def __call__(self, text: str)->[str]:
        text = self.pre_process(text)
        words = self.normal_token(text)
        return words




class baseline_MyDataset(Dataset):
    def __init__(self, dataset_name, data_path, using_bert=False, is_to_tokens=True):
        self.dataset_name = dataset_name
        self.dataset_path = data_path
        self.labels_num = baseline_config_dataset[dataset_name].labels_num
        self.data, self.labels = read_standard_data(data_path)
        self.data_token = []
        self.data_seq = []
        self.labels_tensor = []
        self.vocab = None
        self.using_bert = using_bert
        if using_bert:
            self.tokenizer = BertTokenizer.from_pretrained(baseline_BertConfig.model_name)
            self.data_types = []
            self.data_masks = []
        else:
            self.tokenizer = baseline_Tokenizer()
        self.maxlen = None

        if is_to_tokens:
            self.data2token()


    def data2token(self):
        assert self.data is not None
        logging(f'{self.dataset_name} {self.dataset_path} is tokenizing')
        if self.using_bert:
            pass
        else:
            for sen in tqdm(self.data):
                self.data_token.append(self.tokenizer(sen))

    def token2seq(self, vocab:'Vocab', maxlen:int):
        logging(f'{self.dataset_name} {self.dataset_path} is seq maxlen {maxlen}')
        if not self.using_bert:
            if len(self.data_seq) > 0:
                self.data_seq.clear()
                self.labels_tensor.clear()
            self.vocab = vocab
            self.maxlen = maxlen
            assert self.data_token is not None
            for tokens in self.data_token:
                self.data_seq.append(self.__encode_tokens(tokens))
        else:
            for sen in tqdm(self.data):
                t = self.tokenizer(sen, max_length=maxlen, truncation=True, padding=True)
                self.data_token.append(torch.tensor(t['input_ids'], dtype=torch.long))
                self.data_types.append(torch.tensor(t['token_type_ids'], dtype=torch.long))
                self.data_masks.append(torch.tensor(t['attention_mask'], dtype=torch.long))
                assert self.data_token[-1].size()[0] == self.data_types[-1].size()[0] == self.data_masks[-1].size()[0]
                print(self.data_token[-1].size(), len(self.data_token))
            self.data_seq = self.data_token
        for label in self.labels:
            self.labels_tensor.append(torch.tensor(label))

    def __encode_tokens(self, tokens)->torch.Tensor:
        '''
        if one sentence length is shorter than maxlen, it will use pad word for padding to maxlen
        :param tokens:
        :return:
        '''
        pad_word = 0
        x = [pad_word for _ in range(self.maxlen)]
        temp = tokens[:self.maxlen]
        for idx, word in enumerate(temp):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    def split_data_by_label(self):
        datas = [[] for _ in range(self.labels_num)]
        for idx, lab in enumerate(self.labels):
            temp = (self.data[idx], lab)
            datas[lab].append(temp)
        return datas

    def sample_by_labels(self, single_label_num:int):
        datas = self.split_data_by_label()
        sample_data = []
        sample_label = [-1 for _ in range(single_label_num*self.labels_num)]
        for i in range(self.labels_num):
            sample_data += random.sample(datas[i], single_label_num)
        for idx, data in enumerate(sample_data):
            sample_data[idx] = data[0]
            sample_label[idx] = data[1]
        assert len(sample_data) == len(sample_label)
        return sample_data, sample_label


    def statistic(self):
        import numpy as np
        length = [len(x) for x in self.data_token]
        logging(f'statistic {self.dataset_name}'
              f'maxlen {max(length)}, minlen {min(length)}, '
              f'meanlen {sum(length) / len(length)}, medianlen {np.median(length)}')

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, item):
        if self.using_bert:
            return (self.data_seq[item], self.data_types[item], self.data_masks[item],
                    self.labels_tensor[item])
        return (self.data_seq[item], self.labels_tensor[item])


if __name__ == '__main__':
    pass
    tokenizer = BertTokenizer.from_pretrained(baseline_BertConfig.model_name)
    text ='hello, i love u.. you and me'

    print(tokenizer(text, max_length=1))