import random
import re
import spacy
from spacy.lang.en import English
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer
from baseline_config import baseline_config_dataset, baseline_BertConfig
from baseline_tools import logging, read_standard_data, read_SNLI_origin_data


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

class spacy_Tokenizer():
    def __init__(self):
        nlp = English()
        self.tokenizer = nlp.tokenizer

    def __call__(self, text:str)->[str]:
        return [t.text for t in self.tokenizer(text)]

class baseline_MyDataset(Dataset):
    def __init__(self, dataset_name, data_path, using_bert=False, is_to_tokens=True):
        self.dataset_name = dataset_name
        self.dataset_path = data_path
        self.dataset_config = baseline_config_dataset[dataset_name]
        self.labels_num = self.dataset_config.labels_num
        if dataset_name == 'SNLI':
            self.premise_data, self.hypothesis_data, self.labels = read_SNLI_origin_data(
                baseline_config_dataset[dataset_name].sentences_path, data_path
            )
            self.data_token = {
                'pre': [],
                'hypo': [],
                'comb': [], # bert will use the combined one
            }
            self.data_tensor = {
                'pre': [],
                'pre_len': [],
                'hypo': [],
                'hypo_len': [],
                'comb': [],
            }
        else:
            self.data, self.labels = read_standard_data(data_path)
            self.data_token = []
            self.data_tensor = []
        self.labels_tensor = []
        self.vocab = None
        self.using_bert = using_bert
        if using_bert:
            self.tokenizer = BertTokenizer.from_pretrained(baseline_BertConfig.model_name)
            self.data_types = []
            self.data_masks = []
        elif self.dataset_config.tokenizer_type == 'normal':
            self.tokenizer = baseline_Tokenizer()
        elif self.dataset_config.tokenizer_type == 'spacy':
            self.tokenizer = spacy_Tokenizer()
        else:
            raise KeyError(f'tokenizer {self.dataset_config.tokenizer_type} not supported')
        self.maxlen = None

        if is_to_tokens:
            self.data2token()


    def data2token(self):
        logging(f'{self.dataset_name} {self.dataset_path} is tokenizing')
        if self.using_bert:
            pass
        elif self.dataset_name == 'SNLI':
            with tqdm(total=len(self.premise_data)+len(self.hypothesis_data)) as pbar:
                for sen in self.premise_data:
                    self.data_token['pre'].append(self.tokenizer(sen))
                    pbar.update(1)
                for sen in self.hypothesis_data:
                    self.data_token['hypo'].append(self.tokenizer(sen))
                    pbar.update(1)
        else:
            for sen in tqdm(self.data):
                self.data_token.append(self.tokenizer(sen))

    def token2seq(self, vocab:'Vocab', maxlen:int):
        logging(f'{self.dataset_name} {self.dataset_path} is seq maxlen {maxlen}')
        if not self.using_bert:
            self.vocab = vocab
            self.maxlen = maxlen
            if self.dataset_name == 'SNLI':
                assert len(self.data_token['pre']) == len(self.data_token['hypo']) == len(self.labels)
                for tokens in self.data_token['pre']:
                    s_len = min(len(tokens), maxlen)
                    self.data_tensor['pre_len'].append(torch.tensor(s_len, dtype=torch.long))
                    self.data_tensor['pre'].append(self.__encode_tokens(tokens))
                for tokens in self.data_token['hypo']:
                    s_len = min(len(tokens), maxlen)
                    self.data_tensor['hypo_len'].append(torch.tensor(s_len, dtype=torch.long))
                    self.data_tensor['hypo'].append(self.__encode_tokens(tokens))
            else:
                for tokens in self.data_token:
                    self.data_tensor.append(self.__encode_tokens(tokens))
        else:
            if self.dataset_name == 'SNLI':
                for i in tqdm(range(len(self.premise_data))):
                    t = self.tokenizer(text=self.premise_data[i], text_pair=self.hypothesis_data[i],
                                       max_length=maxlen*2, truncation=True, padding='max_length')
                    self.data_token['comb'].append(torch.tensor(t['input_ids'], dtype=torch.long))
                    self.data_types.append(torch.tensor(t['token_type_ids'], dtype=torch.long))
                    self.data_masks.append(torch.tensor(t['attention_mask'], dtype=torch.long))
            else:
                for sen in tqdm(self.data):
                    t = self.tokenizer(sen, max_length=maxlen, truncation=True, padding='max_length')
                    self.data_token.append(torch.tensor(t['input_ids'], dtype=torch.long))
                    self.data_types.append(torch.tensor(t['token_type_ids'], dtype=torch.long))
                    self.data_masks.append(torch.tensor(t['attention_mask'], dtype=torch.long))
                self.data_tensor = self.data_token

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


    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, item):
        if self.using_bert:
            if self.dataset_name == 'SNLI':
                return (self.data_token['comb'][item], self.data_types[item], self.data_masks[item],
                        self.labels_tensor[item])
            return (self.data_tensor[item], self.data_types[item], self.data_masks[item],
                    self.labels_tensor[item])
        if self.dataset_name == 'SNLI':
            return ((self.data_tensor['pre'][item], self.data_tensor['pre_len'][item]),
                    (self.data_tensor['hypo'][item], self.data_tensor['hypo_len'][item]), self.labels_tensor[item])
        return (self.data_tensor[item], self.labels_tensor[item])


if __name__ == '__main__':
    pass
    # tokenizer = BertTokenizer.from_pretrained(baseline_BertConfig.model_name)
    # text1 = 'A boy is jumping on skateboard in the middle of a red bridge .'
    # text2 = 'An older man sits with his orange juice at a small table in a coffee shop while employee'
    # res = tokenizer(text=text1, text_pair=text2, max_length=10*2, truncation=True, padding='max_length')
    # res = tokenizer.convert_ids_to_tokens(res['input_ids'])
    # print(res)

