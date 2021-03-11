import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
import os
import sys
import numpy as np
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
from baseline_config import baseline_BertConfig


class baseline_LSTM(nn.Module):

    def __init__(self,
                 num_hiddens:int, num_layers:int, word_dim:int, vocab:'Vocab', labels_num:int,
                 using_pretrained=True, bid=False, head_tail=False):
        super(baseline_LSTM, self).__init__()
        if bid:
            self.model_name = 'BidLSTM'
        else:
            self.model_name = 'LSTM'
        self.head_tail = head_tail
        self.bid = bid

        self.embedding_layer = nn.Embedding(vocab.num, word_dim)
        self.embedding_layer.weight.requires_grad = True
        if using_pretrained:
            assert vocab.word_dim == word_dim
            assert vocab.num == vocab.vectors.shape[0]
            self.embedding_layer.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding_layer.weight.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.encoder = nn.LSTM(
            input_size=word_dim, hidden_size=num_hiddens,
            num_layers=num_layers, bidirectional=bid,
            dropout=0.3

        )

        # using bidrectional, *2
        if bid:
            num_hiddens *= 2
        if head_tail:
            num_hiddens *= 2


        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_hiddens, labels_num),
        )


    def forward(self, X: torch.Tensor, types=None, masks=None):
        X = X.permute(1, 0) # [batch, seq_len] -> [seq_len, batch]
        X = self.embedding_layer(X)  #[seq_len, batch, word_dim]

        X = self.dropout(X)

        outputs, _ = self.encoder(X) # output, (hidden, memory)
        # outputs [seq_len, batch, hidden*2] *2 means using bidrectional
        # head and tail, [batch, hidden*4]


        temp = torch.cat((outputs[0], outputs[-1]), -1) if self.head_tail else outputs[-1]

        outputs = self.fc(temp) # [batch, hidden*4] -> [batch, labels]

        return outputs

    def forward_with_embedding(self, embeddings: torch.Tensor):
        # [batch, seq, embeddings]
        embeddings = embeddings.permute(1, 0)
        X = self.dropout(embeddings)

        outputs, _ = self.encoder(X) # output, (hidden, memory)
        # outputs [seq_len, batch, hidden*2] *2 means using bidrectional
        # head and tail, [batch, hidden*4]


        temp = torch.cat((outputs[0], outputs[-1]), -1) if self.head_tail else outputs[-1]

        outputs = self.fc(temp) # [batch, hidden*4] -> [batch, labels]

        return outputs

class baseline_TextCNN(nn.Module):
    def __init__(self, vocab:'Vocab', train_embedding_word_dim, is_static, using_pretrained,
                 num_channels:list, kernel_sizes:list, labels_num:int, is_batch_normal:bool):
        super(baseline_TextCNN, self).__init__()
        self.model_name = 'TextCNN'

        self.using_pretrained = using_pretrained
        self.word_dim = train_embedding_word_dim
        if using_pretrained: self.word_dim += vocab.word_dim

        if using_pretrained:
            self.embedding_pre = nn.Embedding(vocab.num, vocab.word_dim)
            self.embedding_pre.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding_pre.weight.requires_grad = not is_static

        self.embedding_train = nn.Embedding(vocab.num, train_embedding_word_dim)

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.convs = nn.ModuleList()

        if is_batch_normal:
            for c, k in zip(num_channels, kernel_sizes):
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=self.word_dim,
                                  out_channels=c,
                                  kernel_size=k),
                        nn.BatchNorm1d(c)
                    )
                )
        else:
            for c, k in zip(num_channels, kernel_sizes):
                self.convs.append(
                    nn.Conv1d(in_channels=self.word_dim,
                              out_channels=c,
                              kernel_size=k)
                )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_channels), labels_num)



    def forward(self, X:torch.Tensor, types=None, masks=None):
        if self.using_pretrained:
            embeddings = torch.cat((
                self.embedding_train(X),
                self.embedding_pre(X),
            ), dim=-1) # [batch, seqlen, word-dim0 + word-dim1]
        else: embeddings = self.embedding_train(X)

        embeddings = self.dropout(embeddings)

        embeddings = embeddings.permute(0, 2, 1) # [batch, dims, seqlen]


        outs = torch.cat(
            [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)

        outs = self.dropout(outs)

        logits = self.fc(outs)
        return logits

    def forward_with_embedding(self, embeddings:torch.Tensor):
        # [batch, seq, embed]
        assert embeddings.dim()[-1] == self.word_dim

        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1) # [batch, dims, seqlen]


        outs = torch.cat(
            [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)

        outs = self.dropout(outs)

        logits = self.fc(outs)
        return logits

class baseline_Bert(nn.Module):
    def __init__(self, label_num:int, is_fine_tuning=True, is_entailment=False):
        super(baseline_Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained(baseline_BertConfig.model_name)
        self.model_name = 'Bert_E' if is_entailment else 'Bert'
        self.hidden_size = 768
        if not is_fine_tuning:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, label_num),
        )


    def forward(self, x, types, masks):
        # inputs = (x, types, masks)
        encoder, pooled = self.bert_model(x, masks, types)[:]
        logits = self.fc(pooled)
        return logits

class baseline_TextCNN_encoder(nn.Module):
    def __init__(self, word_dim, num_channels:list, kernel_sizes:list, is_batch_normal:bool):
        super(baseline_TextCNN_encoder, self).__init__()



        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.convs = nn.ModuleList()
        if is_batch_normal:
            for c, k in zip(num_channels, kernel_sizes):
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels=word_dim,
                                  out_channels=c,
                                  kernel_size=k),
                        nn.BatchNorm1d(c)
                    )
                )
        else:
            for c, k in zip(num_channels, kernel_sizes):
                self.convs.append(
                    nn.Conv1d(in_channels=word_dim,
                              out_channels=c,
                              kernel_size=k)
                )




    def forward(self, embeddings:torch.Tensor):

        outs = torch.cat(
            [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)


        return outs

class baseline_TextCNN_Entailment(nn.Module):
    def __init__(self, vocab:'Vocab', train_embedding_word_dim, is_static, using_pretrained,
                 num_channels:list, kernel_sizes:list, labels_num:int, is_batch_normal:bool):
        super(baseline_TextCNN_Entailment, self).__init__()
        self.model_name = 'TextCNN_E'

        self.using_pretrained = using_pretrained
        self.word_dim = train_embedding_word_dim
        if using_pretrained: self.word_dim += vocab.word_dim

        if using_pretrained:
            self.embedding_pre = nn.Embedding(vocab.num, vocab.word_dim)
            self.embedding_pre.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding_pre.weight.requires_grad = not is_static

        self.embedding_train = nn.Embedding(vocab.num, train_embedding_word_dim)
        self.dropout = nn.Dropout(0.5)

        self.pre_encoder = baseline_TextCNN_encoder(self.word_dim, num_channels, kernel_sizes, is_batch_normal)
        self.hypo_encoder = baseline_TextCNN_encoder(self.word_dim, num_channels, kernel_sizes, is_batch_normal)

        all_hiddens = sum(num_channels) * 2

        self.layers = nn.Sequential(
            nn.Linear(all_hiddens, all_hiddens//3),
            nn.BatchNorm1d(all_hiddens//3),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(all_hiddens//3, labels_num)
        )

    def forward(self, x:tuple, types=None, masks=None):
        res = []
        for temp in x:
            if self.using_pretrained:
                embeddings = torch.cat((
                    self.embedding_train(temp),
                    self.embedding_pre(temp),
                ), dim=-1) # [batch, seqlen, word-dim0 + word-dim1]
            else: embeddings = self.embedding_train(temp)
            embeddings = self.dropout(embeddings)
            embeddings = embeddings.permute(0, 2, 1)  # [batch, dims, seqlen]
            res.append(embeddings)

        outs_pre = self.pre_encoder(res[0])
        outs_hypo = self.hypo_encoder(res[1])
        outs = torch.cat([outs_pre, outs_hypo], dim=1)

        logits = self.layers(outs)

        return logits

class baseline_Infnet(nn.Module):

    def __init__(self, word_dim, lstm_hidden, num_layer, dropout_rate=0.25, pool_type='max'):
        super(baseline_Infnet, self).__init__()
        self.word_emb_dim = word_dim
        self.enc_lstm_dim = lstm_hidden
        self.pool_type = pool_type


        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, num_layer,
                                bidirectional=True, dropout=dropout_rate)


    def forward(self, sent:torch.Tensor, sent_len:torch.Tensor):
        sort = torch.sort(sent_len, descending=True)
        sent_len_sort, idx_sort = sort.values, sort.indices
        idx_reverse = torch.argsort(idx_sort)

        sent = sent.index_select(1, idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sort)
        outs, _ = self.enc_lstm(sent_packed)
        outs, _ = nn.utils.rnn.pad_packed_sequence(outs)

        outs = outs.index_select(1, idx_reverse)

        if self.pool_type == 'mean':
            sent_len = sent_len.clone().type(torch.float).unsqueeze(1)
            emb = torch.sum(outs, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
            return emb
        elif self.pool_type == 'max':
            emb = torch.max(outs, 0).values
            return emb
        else:
            return outs[-1]


class baseline_BidLSTM_entailment(nn.Module):
    def __init__(self, vocab:'Vocab', word_dim, lstm_hidden, num_layer, labels_num, linear_size, dropout_rate=0.25,
                 using_pretrained=True, pool_type='max'):
        super(baseline_BidLSTM_entailment, self).__init__()
        self.model_name = 'BidLSTM_E'

        dropout_rate = 0.0 if num_layer == 1 else dropout_rate

        self.embedding = nn.Embedding(vocab.num, vocab.word_dim)
        self.embedding.weight.requires_grad = True

        if using_pretrained:
            assert vocab.word_dim == word_dim
            self.embedding.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding.weight.requires_grad = False

        self.encoder = baseline_Infnet(word_dim, lstm_hidden, num_layer, dropout_rate=dropout_rate,
                                       pool_type=pool_type)

        self.inputdim = lstm_hidden * 2 * 4

        self.relu = nn.LeakyReLU()


        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.inputdim, linear_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_size, linear_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_size, labels_num),
        )

    def forward(self, x:tuple, types=None, masks=None):
        # ((x1, x1_len), (x2, x2_len))
        batch_size = x[0][0].size()[0]

        temp = self.embedding(x[0][0].permute(1, 0))
        u = self.encoder(temp, x[0][1]).reshape(batch_size, -1)

        temp = self.embedding(x[1][0].permute(1, 0))
        v = self.encoder(temp, x[1][1]).reshape(batch_size, -1)

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        outs = self.fc(features)

        return outs



if __name__ == '__main__':
    maxlen = 10
    batch_size = 4
    word_dim = 300
    num_hidden = 200


    sentence = torch.randn(maxlen, batch_size, word_dim)
    sentence_len = torch.randint(1, maxlen, size=[batch_size])

    net = baseline_Infnet(word_dim, num_hidden, num_layer=2)
    out = net(sentence, sentence_len)