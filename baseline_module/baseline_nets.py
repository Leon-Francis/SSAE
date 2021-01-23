import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

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


    def forward(self, X: torch.Tensor, types, masks):
        X = X.permute(1, 0) # [batch, seq_len] -> [seq_len, batch]
        X = self.embedding_layer(X)  #[seq_len, batch, word_dim]

        X = self.dropout(X)

        outputs, _ = self.encoder(X) # output, (hidden, memory)
        # outputs [seq_len, batch, hidden*2] *2 means using bidrectional
        # head and tail, [batch, hidden*4]

        outputs = torch.cat((outputs[0], outputs[-1]), -1) if self.head_tail else outputs[-1]

        outputs = self.fc(outputs) # [batch, hidden*4] -> [batch, labels]

        return outputs

class baseline_TextCNN(nn.Module):
    def __init__(self, vocab:'Vocab', train_embedding_word_dim, is_static, using_pretrained,
                 num_channels:list, kernel_sizes:list, labels_num:int):
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
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(
                nn.Conv1d(in_channels=self.word_dim,
                          out_channels=c,
                          kernel_size=k)
            )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(num_channels), labels_num)



    def forward(self, X:torch.Tensor, types, masks):
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

class baseline_Bert(nn.Module):
    def __init__(self, label_num:int, is_fine_tuning=True):
        super(baseline_Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained(baseline_BertConfig.model_name)
        self.model_name = 'Bert'
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