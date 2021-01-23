# configuration of baseline models (classification)
import random

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(667)

baseline_config_model_save_path_format = r'./baseline_models/{}/{}_{:.5f}_{}_{}.pt'

baseline_config_model_load_path = {
    'IMDB': {
        'LSTM': r'./baseline_models/IMDB/LSTM_0.88596.pt',
        'BidLSTM': r'./baseline_models/IMDB/BidLSTM_0.88936.pt',
        'TextCNN': r'./baseline_models/IMDB/TextCNN_0.86428.pt',
    },
    'AGNEWS': {
        'LSTM': r'./baseline_models/AGNEWS/LSTM_0.90118.pt',
        'BidLSTM': r'./baseline_models/AGNEWS/BidLSTM_0.91855.pt',
        'TextCNN': r'./baseline_models/AGNEWS/TextCNN_0.91803.pt',
    },
    'SNLI': {
        'LSTM': r'./baseline_models/SNLI/LSTM.pt',
        'BidLSTM': r'./baseline_models/SNLI/BidLSTM.pt',
        'TextCNN': r'./baseline_models/SNLI/TextCNN.pt',
    }
}

class baseline_IMDBConfig():
    train_data_path = r'../dataset/IMDB/aclImdb/train.std'
    test_data_path = r'../dataset/IMDB/aclImdb/test.std'
    pretrained_word_vectors_path = r'../static/glove.6B.100d.txt'
    labels_num = 2
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 300


class baseline_AGNEWSConfig():
    train_data_path = r'../dataset/AGNEWS/train.std'
    test_data_path = r'../dataset/AGNEWS/test.std'
    pretrained_word_vectors_path = r'../static/glove.6B.100d.txt'
    labels_num = 4
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 50


class baseline_TextCNNConfig():
    channel_kernel_size = {
        'IMDB': ([50, 50, 50], [3, 4, 5]),
        'AGNEWS': ([50, 50, 50], [3, 4, 5]),
    }
    is_static = {
        'IMDB': True,
        'AGNEWS': True,
    }
    using_pretrained = {
        'IMDB': True,
        'AGNEWS': True,
    }

    train_embedding_dim = {
        'IMDB': 50,
        'AGNEWS': 50,
    }

class baseline_LSTMConfig():
    num_hiddens = {
        'IMDB': 100,
        'AGNEWS': 100,
    }

    num_layers = {
        'IMDB': 2,
        'AGNEWS': 2,
    }

    is_using_pretrained = {
        'IMDB': True,
        'AGNEWS': True,
    }

    word_dim = {
        'IMDB': 100,
        'AGNEWS': 100,
    }

class baseline_BertConfig():
    model_name = 'bert-base-uncased'
    is_fine_tuning = {
        'IMDB': True,
        'AGNEWS': True,
        'SNLI': True
    }

baseline_config_dataset = {
    'IMDB': baseline_IMDBConfig,
    'AGNEWS': baseline_AGNEWSConfig,
}

baseline_config_models_list = [
    'Bert', 'LSTM', 'BidLSTM', 'TextCNN'
]
