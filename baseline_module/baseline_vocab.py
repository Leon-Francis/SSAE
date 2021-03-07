import numpy as np
import re
import os
from baseline_tools import *


class baseline_Vocab():

    def __init__(self, origin_data_tokens, vocab_limit_size=80000, is_special=False,
                 is_using_pretrained=True, word_vec_file_path=r'./static/glove.6B.100d.txt'):
        assert len(origin_data_tokens) > 0
        self.file_path = word_vec_file_path
        self.word_dim = int(re.findall("\d+d", word_vec_file_path)[0][:-1])
        self.word_dict = {}
        self.word_count = {}
        self.vectors = None
        self.num = 0
        self.data_tokens = []
        self.words_vocab = []
        self.is_special = is_special # enable <cls> and <sep>
        self.special_word_unk = ('<unk>', 0)
        self.special_word_cls = ('[CLS]', 1)
        self.special_word_sep = ('[SEP]', 2)
        self.data_tokens = origin_data_tokens
        self.__build_words_index()
        self.__limit_dict_size(vocab_limit_size)
        if is_using_pretrained:
            logging(f'building word vectors from {self.file_path}')
            self.__read_pretrained_word_vecs()
        logging(f'word vectors has been built! dict size is {self.num}')


    def __build_words_index(self):
        for sen in self.data_tokens:
            for word in sen:
                if word not in self.word_dict:
                    self.word_dict[word] = self.num
                    self.word_count[word] = 1
                    self.num += 1
                else:
                    self.word_count[word] += 1


    def __limit_dict_size(self, vocab_limit_size):
        limit = vocab_limit_size
        word_count_temp = sorted(self.word_count.items(), key=lambda x: x[1], reverse=True)
        count = 1
        self.words_vocab.append(self.special_word_unk[0])
        self.word_count[self.special_word_unk[0]] = int(1e9)
        if self.is_special:
            self.words_vocab += [self.special_word_cls[0], self.special_word_sep[0]]
            self.word_count[self.special_word_cls[0]] = int(1e9)
            self.word_count[self.special_word_sep[0]] = int(1e9)
            count += 2
        temp = {}
        for x, y in word_count_temp:
            if count > limit:
                break
            temp[x] = count
            self.words_vocab.append(x)
            count += 1
        self.word_dict = temp
        self.word_dict[self.special_word_unk[0]] = 0
        if self.is_special:
            self.word_dict[self.special_word_cls[0]] = 1
            self.word_dict[self.special_word_sep[0]] = 2
        self.num = count
        assert self.num == len(self.word_dict) == len(self.words_vocab)
        self.vectors = np.ndarray([self.num, self.word_dim], dtype='float32')

    def __read_pretrained_word_vecs(self):
        num = 0
        word_dict = {}
        word_dict[self.special_word_unk[0]] = num  # unknown word

        temp = self.file_path + '.pkl'
        if os.path.exists(temp):
            word_dict, vectors = load_pkl_obj(temp)
        else:
            if self.is_special:
                word_dict[self.special_word_cls[0]] = 1
                word_dict[self.special_word_sep[0]] = 2
                num += 2
            with open(self.file_path, 'r', encoding='utf-8') as file:
                file = file.readlines()
                vectors = np.ndarray([len(file) + 1 + num, self.word_dim], dtype='float32')
                vectors[0] = np.random.normal(0.0, 0.3, [self.word_dim])  # unk
                if self.is_special:
                    vectors[1] = np.random.normal(0.0, 0.3, [self.word_dim])
                    vectors[2] = np.random.normal(0.0, 0.3, [self.word_dim])
                for line in file:
                    line = line.split()
                    num += 1
                    word_dict[line[0]] = num
                    vectors[num] = np.asarray(line[-self.word_dim:], dtype='float32')

            save_pkl_obj((word_dict, vectors), temp)

        for word, idx in self.word_dict.items():
            if word in word_dict:
                key = word_dict[word]
                self.vectors[idx] = vectors[key]
            else:
                self.vectors[idx] = vectors[0]



    def __len__(self):
        return self.num

    def get_word_count(self, word):
        # word could be int or str
        if isinstance(word, int):
            word = self.get_word(word)
        if word not in self.word_count:
            return 0 # OOV
        return self.word_count[word]


    def get_index(self, word: str):
        if word not in self.word_dict:
            return 0  # unknown word
        return self.word_dict[word]

    def get_word(self, index:int):
        return self.words_vocab[index]

    def get_vec(self, index: int):
        assert self.vectors is not None
        return self.vectors[index]