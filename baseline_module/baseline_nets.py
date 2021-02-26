import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
import os
import sys
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

        self.vec = torch.from_numpy(vocab.vectors)

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
        # [batch, seq, vovab_size]
        self.vec = self.vec.to(embeddings.device)
        embeddings = torch.matmul(embeddings, self.vec)
        embeddings = embeddings.permute(1, 0, 2)
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

class baseline_LSTM_encoder(nn.Module):

    def __init__(self,
                 num_hiddens:int, num_layers:int, word_dim:int, bid=False, head_tail=False):
        super(baseline_LSTM_encoder, self).__init__()
        self.head_tail = head_tail
        self.bid = bid
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
        self.num_hiddens = num_hiddens




    def forward(self, embeddings: torch.Tensor):


        outputs, _ = self.encoder(embeddings) # output, (hidden, memory)
        # outputs [seq_len, batch, hidden*2] *2 means using bidrectional
        # head and tail, [batch, hidden*4]

        outputs = torch.cat((outputs[0], outputs[-1]), -1) if self.head_tail else outputs[-1]

        return outputs

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

class baseline_LSTM_Entailment(nn.Module):
    def __init__(self, num_hiddens:int, num_layers:int, word_dim:int, vocab:'Vocab', labels_num:int,
                 using_pretrained=True, bid=False, head_tail=False):
        super(baseline_LSTM_Entailment, self).__init__()

        if bid:
            self.model_name = 'BidLSTM_E'
        else:
            self.model_name = 'LSTM_E'

        self.embedding_layer = nn.Embedding(vocab.num, word_dim)
        self.embedding_layer.weight.requires_grad = True
        if using_pretrained:
            assert vocab.word_dim == word_dim
            assert vocab.num == vocab.vectors.shape[0]
            self.embedding_layer.from_pretrained(torch.from_numpy(vocab.vectors))
            self.embedding_layer.weight.requires_grad = False
        self.dropout = nn.Dropout(0.5)

        self.premise_encoder = baseline_LSTM_encoder(num_hiddens, num_layers, word_dim, bid, head_tail)
        self.hypothesis_encoder = baseline_LSTM_encoder(num_hiddens, num_layers, word_dim, bid, head_tail)


        all_hiddens_size = self.premise_encoder.num_hiddens+self.hypothesis_encoder.num_hiddens


        layer_size = [all_hiddens_size, 300, 300]
        self.layers = nn.Sequential(nn.Dropout(0.5))


        for i in range(len(layer_size)-1):
            self.layers.add_module(
                'linear'+str(i), nn.Linear(layer_size[i], layer_size[i+1]),
            )
            self.layers.add_module(
                'relu', nn.ReLU(),
            )
            self.layers.add_module(
                'bn'+str(i), nn.BatchNorm1d(layer_size[i+1])
            )

        self.layers.add_module(
            'fc', nn.Linear(layer_size[-1], labels_num)
        )



    def forward(self, x:tuple, types=None, masks=None):
        x_pre, x_hypo = x[0], x[1]
        x_pre, x_hypo = x_pre.permute(1, 0), x_hypo.permute(1, 0)

        embeddings_pre = self.dropout(self.embedding_layer(x_pre))
        embeddings_hypo = self.dropout(self.embedding_layer(x_hypo))

        outs_pre = self.premise_encoder(embeddings_pre)
        outs_hypo = self.hypothesis_encoder(embeddings_hypo)

        outs = torch.cat([outs_pre, outs_hypo], dim=1)

        return self.layers(outs)

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


if __name__ == '__main__':
    pass