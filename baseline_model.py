import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import BertModel
from config import Baseline_LSTMConfig
from config import Baseline_BertConfig
from config import Baseline_CNNConfig
from tools import load_bert_vocab_embedding_vec


class Baseline_Model_Bert_Classification(nn.Module):
    def __init__(self, dataset_config):
        super(Baseline_Model_Bert_Classification, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = Baseline_BertConfig.hidden_size
        if not Baseline_BertConfig.fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, dataset_config.labels_num),
        )
        for params in self.fc.parameters():
            init.normal_(params, mean=0, std=0.01)

    def forward(self, inputs, inputs_mask):
        """forward

        Args:
            inputs (torch.tensor): [batch, seq_len]
            inputs_mask (torch.tensor): [batch, seq_len]

        Returns:
            logits: [batch, 4]
        """
        encoder, pooled = self.bert_model(inputs,
                                          attention_mask=inputs_mask)[:]
        logits = self.fc(pooled)
        return logits


class Baseline_Model_LSTM_Classification(nn.Module):
    def __init__(self, dataset_config, bidirectional):
        super(Baseline_Model_LSTM_Classification, self).__init__()
        self.vocab_size = Baseline_LSTMConfig.vocab_size
        self.embedding_size = Baseline_LSTMConfig.embedding_size
        self.hidden_size = Baseline_LSTMConfig.hidden_size
        self.num_layers = Baseline_LSTMConfig.num_layers
        self.bidirectional = bidirectional
        self.embedding_layer = nn.Embedding(self.vocab_size,
                                            self.embedding_size)
        if Baseline_LSTMConfig.using_pretrained:
            self.embedding_layer.from_pretrained(
                torch.from_numpy(
                    load_bert_vocab_embedding_vec(
                        self.vocab_size, self.embedding_size,
                        Baseline_LSTMConfig.vocab_path,
                        Baseline_LSTMConfig.embedding_path)))
            self.embedding_layer.weight.requires_grad = False
        else:
            self.embedding_layer.weight.requires_grad = True

        self.dropout = nn.Dropout(0.5)

        self.encoder = nn.LSTM(input_size=self.embedding_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               dropout=Baseline_LSTMConfig.dropout)

        hidden_size = self.hidden_size
        if self.bidirectional:
            hidden_size = hidden_size * 2
        if Baseline_LSTMConfig.head_tail:
            hidden_size = hidden_size * 2

        self.fc = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(hidden_size, dataset_config.labels_num))

        for params in self.fc.parameters():
            init.normal_(params, mean=0, std=0.01)

    def forward(self, X):

        X = self.embedding_layer(X)  # [batch, sen_len, word_dim]
        X = self.dropout(X)
        # X: [sen_len, batch, word_dim]
        X = X.permute(1, 0, 2)
        outputs, hidden = self.encoder(X)  # output, (hidden, memory)
        # outputs [sen_len, batch, hidden]
        # outputs [sen_len, batch, hidden*2] *2 means using bidrectional
        if Baseline_LSTMConfig.head_tail:
            outputs = torch.cat((outputs[0], outputs[-1]), -1)
        else:
            outputs = outputs[-1]
        outputs = self.fc(outputs)  # [batch, hidden] -> [batch, labels]

        return outputs


class Baseline_Model_CNN_Classification(nn.Module):
    def __init__(self, dataset_config):
        super(Baseline_Model_CNN_Classification, self).__init__()

        self.vocab_size = Baseline_CNNConfig.vocab_size
        self.embedding_size = Baseline_CNNConfig.embedding_size

        self.word_dim = self.embedding_size
        self.embedding_train = nn.Embedding(self.vocab_size,
                                            self.embedding_size)

        if Baseline_CNNConfig.using_pretrained:
            self.embedding_pre = nn.Embedding(self.vocab_size,
                                              self.embedding_size)
            self.embedding_pre.from_pretrained(
                torch.from_numpy(
                    load_bert_vocab_embedding_vec(
                        self.vocab_size, self.embedding_size,
                        Baseline_CNNConfig.vocab_path,
                        Baseline_CNNConfig.embedding_path)))
            self.embedding_pre.weight.requires_grad = False
            self.word_dim *= 2

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.channel_size = [self.word_dim] + Baseline_CNNConfig.channel_size
        self.kernel_size = Baseline_CNNConfig.kernel_size
        self.convs = nn.ModuleList()
        for i in range(len(self.kernel_size)):
            self.convs.append(
                nn.Conv1d(in_channels=self.channel_size[i],
                          out_channels=self.channel_size[i + 1],
                          kernel_size=self.kernel_size[i]))

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(Baseline_CNNConfig.channel_size),
                            dataset_config.labels_num)
        for params in self.fc.parameters():
            init.normal_(params, mean=0, std=0.01)

    def forward(self, X):

        if Baseline_CNNConfig.using_pretrained:
            embeddings = torch.cat(
                (
                    self.embedding_train(X),
                    self.embedding_pre(X),
                ), dim=-1)  # [batch, seqlen, word-dim0 + word-dim1]
        else:
            embeddings = self.embedding_train(X)

        embeddings = self.dropout(embeddings)

        embeddings = embeddings.permute(0, 2, 1)  # [batch, dims, seqlen]

        outs = torch.cat([
            self.pool(F.relu(conv(embeddings))).squeeze(-1)
            for conv in self.convs
        ],
                         dim=1)

        outs = self.dropout(outs)

        logits = self.fc(outs)
        return logits


class Baseline_Model_Bert_Entailment(nn.Module):
    def __init__(self, dataset_config):
        super(Baseline_Model_Bert_Entailment, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = Baseline_BertConfig.hidden_size
        if not Baseline_BertConfig.fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, dataset_config.labels_num),
        )
        for params in self.fc.parameters():
            init.normal_(params, mean=0, std=0.01)

    def forward(self, inputs, inputs_mask, inputs_type):
        """forward

        Args:
            inputs (torch.tensor): [batch, seq_len]
            inputs_mask (torch.tensor): [batch, seq_len]

        Returns:
            logits: [batch, 4]
        """
        encoder, pooled = self.bert_model(inputs,
                                          attention_mask=inputs_mask,
                                          token_type_ids=inputs_type)[:]
        logits = self.fc(pooled)
        return logits


class Baseline_Model_LSTM_Entailment(nn.Module):
    def __init__(self, dataset_config, bidirectional):
        super(Baseline_Model_LSTM_Entailment, self).__init__()
        self.hidden_size = Baseline_LSTMConfig.hidden_size
        self.num_layers = Baseline_LSTMConfig.num_layers
        self.vocab_size = Baseline_LSTMConfig.vocab_size
        self.bidirectional = bidirectional
        self.embedding_size = Baseline_LSTMConfig.embedding_size

        self.embedding_layer = nn.Embedding(self.vocab_size,
                                            self.embedding_size)
        if Baseline_LSTMConfig.using_pretrained:
            self.embedding_layer.from_pretrained(
                torch.from_numpy(
                    load_bert_vocab_embedding_vec(
                        self.vocab_size, self.embedding_size,
                        Baseline_LSTMConfig.vocab_path,
                        Baseline_LSTMConfig.embedding_path)))
            self.embedding_layer.weight.requires_grad = False

        self.premise_encoder = nn.LSTM(input_size=self.embedding_size,
                                       hidden_size=self.hidden_size,
                                       num_layers=self.num_layers,
                                       bidirectional=self.bidirectional,
                                       dropout=Baseline_LSTMConfig.dropout)

        self.hypothesis_encoder = nn.LSTM(input_size=self.embedding_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_layers,
                                          bidirectional=self.bidirectional,
                                          dropout=Baseline_LSTMConfig.dropout)
        self.layers = nn.Sequential()

        layer_sizes = [2 * self.hidden_size, 400, 100]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.add_module("bn" + str(i + 1), bn)

            self.layers.add_module("activation" + str(i + 1), nn.ReLU())

        layer = nn.Linear(layer_sizes[-1], 3)
        self.layers.add_module("layer" + str(len(layer_sizes)), layer)

        self.layers.add_module("softmax", nn.Softmax(dim=1))

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, premise_indices, hypothesis_indices):
        # premise: [batch, sen_len, embedding_size]
        premise = self.embedding_layer(premise_indices)
        output_prem, (hidden_prem, _) = self.premise_encoder(premise.permute(1, 0, 2))
        # hidden_prem: [batch, hidden_size]
        hidden_prem = hidden_prem[-1]
        if hidden_prem.requires_grad:
            hidden_prem.register_hook(self.store_grad_norm)

        hypothesis = self.embedding_layer(hypothesis_indices)
        output_hypo, (hidden_hypo,
                      _) = self.hypothesis_encoder(hypothesis.permute(1, 0, 2))
        hidden_hypo = hidden_hypo[-1]
        if hidden_hypo.requires_grad:
            hidden_hypo.register_hook(self.store_grad_norm)

        concatenated = torch.cat([hidden_prem, hidden_hypo], 1)
        probs = self.layers(concatenated)
        return probs
