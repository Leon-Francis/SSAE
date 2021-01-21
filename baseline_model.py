import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from transformers import BertModel
from config import Baseline_LSTMConfig
from config import Baseline_BertConfig
from config import Baseline_CNNConfig
from config import AllConfig
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
                               dropout=Baseline_LSTMConfig.dropout,
                               batch_first=True)

        if self.bidirectional:
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.hidden_size * 2, dataset_config.labels_num),
            )
        else:
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.hidden_size, dataset_config.labels_num),
            )
        for params in self.fc.parameters():
            init.normal_(params, mean=0, std=0.01)

    def initHidden(self, batch_size):
        if self.bidirectional:
            return (torch.rand(self.num_layers * 2, batch_size,
                               self.hidden_size).to(AllConfig.train_device),
                    torch.rand(self.num_layers * 2, batch_size,
                               self.hidden_size).to(AllConfig.train_device))
        else:
            return (torch.rand(self.num_layers, batch_size,
                               self.hidden_size).to(AllConfig.train_device),
                    torch.rand(self.num_layers, batch_size,
                               self.hidden_size).to(AllConfig.train_device))

    def forward(self, X):

        X = self.embedding_layer(X)  # [batch, sen_len, word_dim]
        X = self.dropout(X)
        hidden = self.initHidden(X.shape[0])
        outputs, _ = self.encoder(X, hidden)  # output, (hidden, memory)
        # outputs [batch, seq_len, hidden*2] *2 means using bidrectional

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

    def forward(self, X: torch.Tensor):

        if self.using_pretrained:
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
