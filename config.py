import torch

bert_vocab_size = 30522
glove_embedding_size = 100
config_path = './config.py'


class AllConfig():
    output_dir = r'./output'
    cuda_idx = 0
    train_device = torch.device('cuda:' + str(cuda_idx))
    dataset = 'AGNEWS'  # choices = 'IMDB', 'AGNEWS', 'SNLI'
    baseline_model = 'LSTM'  # choices = 'LSTM', 'CNN', 'BidLSTM', 'BERT'
    debug_mode = True
    epochs = 20
    batch_size = 64


class IMDBConfig():
    train_data_path = r'./dataset/IMDB/aclImdb/train'
    test_data_path = r'./dataset/IMDB/aclImdb/test'
    labels_num = 2
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 230
    vocab_size = bert_vocab_size


class AGNEWSConfig():
    train_data_path = r'./dataset/AGNEWS/train.std'
    test_data_path = r'./dataset/AGNEWS/test.std'
    labels_num = 4
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 50
    vocab_size = bert_vocab_size


class SNLIConfig():
    train_data_path = r'./dataset/SNLI/train.txt'
    test_data_path = r'./dataset/SNLI/test.txt'
    sentences_data_path = r'./dataset/SNLI/sentences.txt'
    labels_num = 3
    label_classes = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    tokenizer_type = 'bert'
    remove_stop_words = False
    sen_len = 10
    vocab_size = bert_vocab_size


class Baseline_LSTMConfig():
    vocab_size = bert_vocab_size
    embedding_size = glove_embedding_size
    vocab_path = r'./static/vocab.txt'
    embedding_path = r'./static/glove.6B.100d.txt'
    hidden_size = 100
    num_layers = 2
    using_pretrained = True
    dropout = 0.3
    learning_rate = {'IMDB': 1e-3, 'AGNEWS': 1e-3, 'SNLI': 1e-3}


class Baseline_CNNConfig():
    vocab_size = bert_vocab_size
    embedding_size = glove_embedding_size
    vocab_path = r'./static/vocab.txt'
    embedding_path = r'./static/glove.6B.100d.txt'
    channel_size = [100, 100, 100]
    kernel_size = [3, 4, 5]
    using_pretrained = True
    dropout = 0.3
    learning_rate = {'IMDB': 1e-3, 'AGNEWS': 1e-3, 'SNLI': 1e-3}


class Baseline_BertConfig():
    vocab_size = bert_vocab_size
    hidden_size = 768
    fine_tuning = False
    learning_rate = {'IMDB': 1e-3, 'AGNEWS': 1e-3, 'SNLI': 1e-3}


# class Baseline_TestConfig():

dataset_config_data = {
    'IMDB': IMDBConfig,
    'AGNEWS': AGNEWSConfig,
    'SNLI': SNLIConfig,
}

baseline_model_config_data = {
    'LSTM': Baseline_LSTMConfig,
    'BidLSTM': Baseline_LSTMConfig,
    'CNN': Baseline_CNNConfig,
    'BERT': Baseline_BertConfig
}

baseline_model_lists = ['LSTM', 'CNN', 'BidLSTM', 'BERT']

dataset_list = [
    'IMDB',
    'AGNEWS',
    'SNLI',
]
