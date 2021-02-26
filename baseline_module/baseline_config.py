# configuration of baseline models (classification)
import random
from baseline_tools import logging
import numpy as np
import torch

absolute_path_prefix = r'/home/jsjlab/projects/AttackViaGan'


def setup_seed(seed):
    logging(f'setup seed {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(667)

baseline_config_model_save_path_format = absolute_path_prefix+r'/baseline_module/baseline_models/{}/{}_{:.5f}_{}_{}.pt'

baseline_config_model_load_path = {
    'IMDB': {
        'LSTM': absolute_path_prefix+r'/baseline_module/baseline_models/IMDB/LSTM_0.86532_01_24-13-05_.pt',
        'BidLSTM': absolute_path_prefix+r'/baseline_module/baseline_models/IMDB/BidLSTM_0.86444_01_24-14-12_.pt',
        'TextCNN': absolute_path_prefix+r'/baseline_module/baseline_models/IMDB/TextCNN_0.86736_01_24-15-01_.pt',
        'Bert': absolute_path_prefix+r'/baseline_module/baseline_models/IMDB/Bert_0.91652_01_23-23-59_.pt',
    },
    'AGNEWS': {
        'LSTM': absolute_path_prefix+r'/baseline_module/baseline_models/AGNEWS/LSTM_0.90118_.pt',
        'BidLSTM': absolute_path_prefix+r'/baseline_module/baseline_models/AGNEWS/BidLSTM_0.91855_.pt',
        'TextCNN': absolute_path_prefix+r'/baseline_module/baseline_models/AGNEWS/TextCNN_0.91803_.pt',
        'Bert': absolute_path_prefix+r'/baseline_module/baseline_models/AGNEWS/Bert_0.94132_01_24-01-24_.pt',
    },
    'SNLI': {
        'LSTM_E': absolute_path_prefix+r'/baseline_module/baseline_models/SNLI/LSTM_E_0.74338_01_29-23-40_.pt',
        'BidLSTM_E': absolute_path_prefix+r'/baseline_module/baseline_models/SNLI/BidLSTM_E_0.75214_01_30-15-30_.pt',
        'TextCNN_E': absolute_path_prefix+r'/baseline_module/baseline_models/SNLI/TextCNN_E_0.74287_01_29-14-46_.pt',
        'Bert_E': absolute_path_prefix+r'/baseline_module/baseline_models/SNLI/Bert.pt',
    }
}

class baseline_IMDBConfig():
    train_data_path = absolute_path_prefix+r'/dataset/IMDB/aclImdb/train.std'
    test_data_path = absolute_path_prefix+'/dataset/IMDB/aclImdb/test.std'
    pretrained_word_vectors_path = absolute_path_prefix+r'/static/glove.6B.100d.txt'
    labels_num = 2
    vocab_limit_size = 40000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 230

class baseline_AGNEWSConfig():
    train_data_path = absolute_path_prefix+r'/dataset/AGNEWS/train.std'
    test_data_path = absolute_path_prefix+r'/dataset/AGNEWS/test.std'
    pretrained_word_vectors_path = absolute_path_prefix+r'/static/glove.6B.100d.txt'
    labels_num = 4
    vocab_limit_size = 80000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 50

class baseline_SNLIConfig():
    train_data_path = absolute_path_prefix+r'/dataset/SNLI/train.txt'
    test_data_path = absolute_path_prefix+r'/dataset/SNLI/test.txt'
    sentences_path = absolute_path_prefix+r'/dataset/SNLI/sentences.txt'
    pretrained_word_vectors_path = absolute_path_prefix+r'/static/glove.6B.100d.txt'
    labels_num = 3
    vocab_limit_size = 30000
    tokenizer_type = 'normal'
    remove_stop_words = False
    padding_maxlen = 10

class baseline_TextCNNConfig():
    channel_kernel_size = {
        'IMDB': ([50, 50, 50], [3, 4, 5]),
        'AGNEWS': ([50, 50, 50], [3, 4, 5]),
        'SNLI': ([30, 40, 50, 60], [2, 3, 4, 5])
    }
    is_static = {
        'IMDB': True,
        'AGNEWS': True,
        'SNLI': True,
    }
    using_pretrained = {
        'IMDB': True,
        'AGNEWS': True,
        'SNLI': True
    }

    train_embedding_dim = {
        'IMDB': 50,
        'AGNEWS': 50,
        'SNLI': 100
    }

    is_batch_normal = {
        'IMDB': True,
        'AGNEWS': False,
        'SNLI': True
    }

class baseline_LSTMConfig():
    num_hiddens = {
        'IMDB': 128,
        'AGNEWS': 100,
        'SNLI': 256,
    }

    num_layers = {
        'IMDB': 2,
        'AGNEWS': 2,
        'SNLI': 2,
    }

    is_using_pretrained = {
        'IMDB': False,
        'AGNEWS': True,
        'SNLI': False,
    }

    word_dim = {
        'IMDB': 50,
        'AGNEWS': 100,
        'SNLI': 300
    }

    is_head_and_tail = {
        'IMDB': False,
        'AGNEWS': True,
        'SNLI': True
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
    'SNLI': baseline_SNLIConfig,
}

baseline_config_models_list = [
    'Bert', 'LSTM', 'BidLSTM', 'TextCNN',
    'Bert_E', 'LSTM_E', 'BidLSTM_E', 'TextCNN_E'
]
