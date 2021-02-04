import torch

bert_vocab_size = 30522
glove_embedding_size = 100
config_path = './config.py'


class AttackConfig():
    output_dir = r'./output'
    train_multi_cuda = False
    cuda_idx = 0
    if train_multi_cuda:
        multi_cuda_idx = [0, 1, 2, 3]
        cuda_idx = multi_cuda_idx[0]
    train_device = torch.device('cuda:' + str(cuda_idx))
    dataset = 'AGNEWS'  # choices = 'IMDB', 'AGNEWS', 'SNLI'
    baseline_model = 'LSTM'  # choices = 'LSTM', 'TextCNN', 'BidLSTM', 'Bert'
    debug_mode = False
    epochs = 60
    batch_size = 128

    load_pretrained_Seq2Seq = True
    head_tail = False
    fine_tuning = False
    limit_gen_train = False
    gan_model = 'MLP'  # choices = 'MLP', 'LSTM'

    if load_pretrained_Seq2Seq:
        if dataset == 'AGNEWS':
            pretrained_Seq2Seq_path = r'./output/seq2seq_model/AGNEWS/1612375424/Seq2Seq_model.pt'
            pretrained_huffman_tree_path = r'./output/seq2seq_model/AGNEWS/1612375424/Huffman_tree.pt'

    if fine_tuning:
        Seq2Seq_learning_rate_BERT = 5e-6
        Seq2Seq_learning_rate = 1e-4
    else:
        Seq2Seq_learning_rate = 1e-4
    gan_gen_learning_rate = 1e-4
    gan_adv_learning_rate = 1e-4

    hidden_size = 768
    num_layers = 3
    dropout = 0.3
    super_hidden_size = 500
    vocab_size = bert_vocab_size

    gan_schedule = [1, 3, 5]
    seq2seq_train_times = 1
    gan_gen_train_times = 1
    gan_adv_train_times = 1
    if limit_gen_train:
        gan_gen_train_limit = 7
    else:
        gan_gen_train_limit = epochs

    perturb_sample_num = 20
    perturb_search_times = 1
    perturb_search_bound = 0.005


class BaselineConfig():
    output_dir = r'./output'
    cuda_idx = 1
    train_device = torch.device('cuda:' + str(cuda_idx))
    dataset = 'AGNEWS'  # choices = 'IMDB', 'AGNEWS', 'SNLI'
    baseline_model = 'LSTM'  # choices = 'LSTM', 'TextCNN', 'BidLSTM', 'Bert'
    debug_mode = False
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
    head_tail = True
    dropout = 0.3
    learning_rate = {'IMDB': 1e-3, 'AGNEWS': 1e-3, 'SNLI': 1e-3}


class Baseline_CNNConfig():
    vocab_size = bert_vocab_size
    embedding_size = glove_embedding_size
    vocab_path = r'./static/vocab.txt'
    embedding_path = r'./static/glove.6B.100d.txt'
    channel_size = [200, 200, 200]
    kernel_size = [3, 4, 5]
    using_pretrained = True
    dropout = 0.3
    learning_rate = {'IMDB': 1e-3, 'AGNEWS': 1e-3, 'SNLI': 1e-3}


class Baseline_BertConfig():
    vocab_size = bert_vocab_size
    hidden_size = 768
    fine_tuning = False
    learning_rate = {'IMDB': 1e-3, 'AGNEWS': 1e-3, 'SNLI': 1e-3}


dataset_config_data = {
    'IMDB': IMDBConfig,
    'AGNEWS': AGNEWSConfig,
    'SNLI': SNLIConfig,
}

baseline_model_config_data = {
    'LSTM': Baseline_LSTMConfig,
    'BidLSTM': Baseline_LSTMConfig,
    'TextCNN': Baseline_CNNConfig,
    'Bert': Baseline_BertConfig
}

baseline_model_lists = ['LSTM', 'TextCNN', 'BidLSTM', 'Bert']

dataset_list = [
    'IMDB',
    'AGNEWS',
    'SNLI',
]

baseline_model_load_path = {
    'IMDB': {
        'LSTM':
        r'./output/baseline_model/IMDB/LSTM/1611247993/baseline_model.pt',
        'TextCNN':
        r'./output/baseline_model/IMDB/TextCNN/1611248030/baseline_model.pt',
        'BidLSTM':
        r'./output/baseline_model/IMDB/BidLSTM/1611248060/baseline_model.pt',
        'Bert':
        r'./output/baseline_model/IMDB/Bert/1611248987/baseline_model.pt',
    },
    'AGNEWS': {
        'LSTM':
        r'./output/baseline_model/AGNEWS/LSTM/1611246841/baseline_model.pt',
        'BidLSTM':
        r'./output/baseline_model/AGNEWS/BidLSTM/1611246875/baseline_model.pt',
        'TextCNN':
        r'./output/baseline_model/AGNEWS/TextCNN/1611246902/baseline_model.pt',
        'Bert':
        r'./output/baseline_model/AGNEWS/Bert/1611246760/baseline_model.pt',
    },
    'SNLI': {}
}
