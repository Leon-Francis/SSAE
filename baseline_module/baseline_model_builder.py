import torch
import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from baseline_config import baseline_config_dataset, baseline_LSTMConfig, baseline_TextCNNConfig, \
    baseline_BertConfig, baseline_config_models_list
from baseline_config import baseline_config_model_load_path
from baseline_data import baseline_MyDataset
from baseline_nets import baseline_LSTM, baseline_TextCNN, baseline_Bert, \
    baseline_BidLSTM_entailment, baseline_TextCNN_Entailment
from baseline_tools import logging
from baseline_vocab import baseline_Vocab


class BaselineModelBuilder():
    def __init__(self, dataset_name, model_name, device:torch.device, is_load=True, vocab=None):
        assert dataset_name in {'AGNEWS', 'IMDB', 'SNLI'}
        assert model_name in baseline_config_models_list

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.dataset_config = baseline_config_dataset[self.dataset_name]
        self.device = device
        self.net = None
        self.mode_is_training = False
        if 'Bert' not in model_name:
            self.vocab = vocab if vocab else self.__build_vocab()
        else:
            self.vocab = None

        is_entailment = dataset_name == 'SNLI'
        if model_name in {'LSTM', 'LSTM_E'}:
            self.net = self.__build_LSTM(is_bid=False, is_load=is_load, is_entailment=is_entailment)
        elif 'BidLSTM' in model_name:
            self.net = self.__build_LSTM(is_bid=True, is_load=is_load, is_entailment=is_entailment)
        elif 'TextCNN' in model_name:
            self.net = self.__build_TextCNN(is_load=is_load, is_entailment=is_entailment)
        elif 'Bert' in model_name:
            self.net = self.__build_Bert(is_load=is_load, is_entailment=is_entailment)

        self.net.to(device)
        self.net.eval()

        logging(f'is_load {is_load} loading baseline {self.dataset_name} {self.model_name}')



    def set_training_mode(self):
        self.mode_is_training = True
        self.net.train()

    def set_eval_mode(self):
        self.mode_is_training = False
        self.net.eval()

    def __call__(self, X, types=None, masks=None):
        return self.net.forward(X, types, masks)

    @torch.no_grad()
    def predict_prob(self, X: torch.Tensor, y_true: torch, types=None, masks=None) -> [float]:
        # return the probability of true_label
        if self.mode_is_training:
            raise RuntimeError('you shall take the model in eval to get probability!')

        X, y_true = X.to(self.device), y_true.to(self.device)
        if X.dim() == 1: X = X.view(1, -1)
        if y_true.dim() == 0:  y_true = y_true.view(1)
        if isinstance(types, torch.Tensor) and isinstance(masks, torch.Tensor):
            types, masks = types.to(self.device), masks.to(self.device)
            if types.dim() == 1: types = types.view(1, -1)
            if masks.dim() == 1: masks = masks.view(1, -1)
            logits = self(X, types=types, masks=masks)
        else:
            logits = self(X)

        prob = [logits[i][y_true[i]].item() for i in range(y_true.size(0))]
        return prob

    @torch.no_grad()
    def predict_class(self, X: torch.Tensor, types=None, masks=None) -> [int]:
        # return the predict class number

        if self.mode_is_training:
            raise RuntimeError('you shall take the model in eval to get probability!')

        X = X.to(self.device)
        if X.dim() == 1: X = X.view(1, -1)
        if isinstance(types, torch.Tensor) and isinstance(masks, torch.Tensor):
            types, masks = types.to(self.device), masks.to(self.device)
            if types.dim() == 1: types = types.view(1, -1)
            if masks.dim() == 1: masks = masks.view(1, -1)
            logits = self(X, types=types, masks=masks)
        else:
            logits = self(X)

        predicts = [one.argmax(0).item() for one in logits]
        return predicts

    def __build_vocab(self):
        logging(f'{self.dataset_name} {self.model_name} is building vocab')
        train_data_path = self.dataset_config.train_data_path
        train_dataset = baseline_MyDataset(self.dataset_name, train_data_path, is_to_tokens=True)

        if self.dataset_name == 'SNLI':
            return baseline_Vocab(train_dataset.data_token['pre']+train_dataset.data_token['hypo'],
                                  is_using_pretrained=True, is_special=True,
                                  vocab_limit_size=self.dataset_config.vocab_limit_size,
                                  word_vec_file_path=self.dataset_config.pretrained_word_vectors_path)
        return baseline_Vocab(train_dataset.data_token, is_using_pretrained=True, is_special=False,
                                    vocab_limit_size=self.dataset_config.vocab_limit_size,
                                    word_vec_file_path=self.dataset_config.pretrained_word_vectors_path)

    def __build_LSTM(self, is_bid, is_load, is_entailment):
        num_hiddens = baseline_LSTMConfig.num_hiddens[self.dataset_name]
        num_layers = baseline_LSTMConfig.num_layers[self.dataset_name]
        is_using_pretrained = baseline_LSTMConfig.is_using_pretrained[self.dataset_name]
        word_dim = baseline_LSTMConfig.word_dim[self.dataset_name]
        is_head_and_tail = baseline_LSTMConfig.is_head_and_tail[self.dataset_name]
        if is_entailment:
            assert is_bid
            linear_size = baseline_LSTMConfig.linear_size[self.dataset_name]
            dropout_rate = baseline_LSTMConfig.dropout_rate[self.dataset_name]
            pool_type = baseline_LSTMConfig.pool_type[self.dataset_name]
            net = baseline_BidLSTM_entailment(word_dim=word_dim, vocab=self.vocab, lstm_hidden=num_hiddens,
                                              num_layer=num_layers, labels_num=self.dataset_config.labels_num,
                                              dropout_rate=dropout_rate, using_pretrained=True,
                                              linear_size=linear_size, pool_type=pool_type)
        else:
            net = baseline_LSTM(num_hiddens=num_hiddens, num_layers=num_layers, word_dim=word_dim, bid=is_bid,
                                head_tail=is_head_and_tail, vocab=self.vocab, labels_num=self.dataset_config.labels_num,
                                using_pretrained=is_using_pretrained)
        if is_load:
            model_path = baseline_config_model_load_path[self.dataset_name][self.model_name]
            net.load_state_dict(torch.load(model_path, map_location=self.device))

        return net


    def __build_TextCNN(self, is_load, is_entailment):
        channel_size, kernel_size = baseline_TextCNNConfig.channel_kernel_size[self.dataset_name]
        is_static = baseline_TextCNNConfig.is_static[self.dataset_name]
        train_embedding_dim = baseline_TextCNNConfig.train_embedding_dim[self.dataset_name]
        is_batch_normal = baseline_TextCNNConfig.is_batch_normal[self.dataset_name]
        using_pretrained = baseline_TextCNNConfig.using_pretrained[self.dataset_name]

        if is_entailment:
            net = baseline_TextCNN_Entailment(vocab=self.vocab, train_embedding_word_dim=train_embedding_dim,
                                              is_static=is_static, using_pretrained=using_pretrained,
                                              num_channels=channel_size, kernel_sizes=kernel_size,
                                              labels_num=self.dataset_config.labels_num, is_batch_normal=is_batch_normal)
        else:
            net = baseline_TextCNN(vocab=self.vocab, train_embedding_word_dim=train_embedding_dim,
                                                  is_static=is_static, using_pretrained=using_pretrained,
                                                  num_channels=channel_size, kernel_sizes=kernel_size,
                                                  labels_num=self.dataset_config.labels_num, is_batch_normal=is_batch_normal)
        if is_load:
            model_path = baseline_config_model_load_path[self.dataset_name][self.model_name]
            net.load_state_dict(torch.load(model_path, map_location=self.device))

        return net

    def __build_Bert(self, is_load, is_entailment):
        is_fine_tune = baseline_BertConfig.is_fine_tuning[self.dataset_name]
        net = baseline_Bert(self.dataset_config.labels_num, is_fine_tuning=is_fine_tune, is_entailment=is_entailment)

        if is_load:
            model_path = baseline_config_model_load_path[self.dataset_name][self.model_name]
            net.load_state_dict(torch.load(model_path, map_location=self.device))
        return net
