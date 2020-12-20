import torch
import torch.nn as nn
from config import Config
from transformers import BertModel, BertConfig

from transformers import BertTokenizer
from torch import optim


class Seq2Seq_bert(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 noise_std=0.2):
        super(Seq2Seq_bert, self).__init__()
        self.seq_len = Config.sen_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.noise_std = noise_std
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.encoder = BertModel.from_pretrained('bert-base-uncased',
                                                 config=self.bert_config)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.embedding_decoder = nn.Embedding(Config.vocab_size,
                                              self.embedding_size)
        self.decoder = nn.LSTM(input_size=Config.hidden_size * 2,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout=self.dropout)
        self.mlp_decoder = nn.Linear(self.hidden_size, Config.vocab_size)

    def encode(self, inputs, inputs_mask, is_noise=False):
        """bert_based_encode

        Args:
            inputs (torch.tensor): origin input # [batch, seq_len]
            inputs_mask (torch.Tensor): origin mask # [batch, seq_len]
            is_noise (bool, optional): whether to add noise. Defaults to False.

        Returns:
            torch.tensor: hidden # [batch, seq_len, hidden_size]
        """
        encoders, pooled = self.encoder(inputs,
                                        attention_mask=inputs_mask,
                                        output_hidden_states=True)
        # pooled [batch, hidden_size]
        # hidden [batch, seq_len, hidden_size]
        hidden = encoders[-1]
        state = encoders[0]
        if is_noise:
            gaussian_noise = torch.normal(mean=torch.zeros_like(hidden),
                                          std=self.noise_std)
            gaussian_noise.to(Config.train_device)
            hidden += gaussian_noise
        return hidden, state

    def decode(self, hidden, state):
        """lstm_based_decode

        Args:
            hidden (torch.tensor): bert_hidden[-1] [batch, seq_len, hidden_size]
            state (torch.tensoor): bert_hidden[0] [batch,seq_len, hidden_size]

        Returns:
            [torch.tensor]: outputs [batch, seq_len, vocab_size]
        """
        # hidden [batch, seq_len, hidden_size]
        # state [batch, seq_len, hidden_size]
        all_hidden = torch.cat([state, hidden], 2)
        # all_hidden [batch, seq_len, hidden_size * 2]
        outputs, _ = self.decoder(all_hidden.permute(1, 0, 2))
        outputs = self.mlp_decoder(outputs).permute(1, 0, 2)
        # outputs [batch, seq_len, vocab_size]
        return outputs

    def forward(self,
                inputs,
                inputs_mask,
                is_noise=False,
                encode_only=False,
                generator=None,
                inverter=None):
        """forward

        Args:
            inputs (torch.tensor): orginal inputs [batch, seq_len]
            inputs_mask (torch.tensor):  orginal mask [batch, seq_len]
            is_noise (bool, optional): whether add noise. Defaults to False.
            encode_only (bool, optional):  Defaults to False.
            generator (func, optional):  Defaults to None.
            inverter (func, optional):  Defaults to None.

        Returns:
            torch.tensor: outputs [batch, seq_len, vocab_size]
        """
        hidden, state = self.encode(inputs,
                                    inputs_mask=inputs_mask,
                                    is_noise=is_noise)
        if encode_only:
            return hidden
        if not generator:
            decoded = self.decode(hidden, state)
        else:
            z_hat = inverter(hidden)
            c_hat = generator(z_hat)
            decoded = self.decode(c_hat, state)
        return decoded


class Seq2SeqAE(nn.Module):
    def __init__(self,
                 vocab_size,
                 seq_len,
                 embedding_size,
                 hidden_size,
                 num_layer=1,
                 dropout=0.0,
                 conv_kernels=[5, 5, 3],
                 conv_strides=[1, 1, 1],
                 conv_in_channels=[500, 700, 1000],
                 noise_std=0.2):
        super(Seq2SeqAE, self).__init__()
        self.noise_std = noise_std
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.conv_in_channels = [self.embedding_size] + conv_in_channels

        self.embedding_layer = nn.Embedding(self.vocab_size,
                                            self.embedding_size,
                                            padding_idx=0)
        self.embedding_decoder_layer = nn.Embedding(self.vocab_size,
                                                    self.embedding_size,
                                                    padding_idx=0)
        self.embedding_layer.weight.data[0].fill_(0)
        self.embedding_decoder_layer.weight.data[0].fill_(0)
        self.embedding_layer.weight.requires_grad = True
        self.embedding_decoder_layer.weight.requires_grad = True

        self.encoder = nn.Sequential()
        temp = torch.randn([1, self.embedding_size, self.seq_len
                            ])  # to define the mlp input size automatically
        for i in range(len(self.conv_in_channels) - 1):
            self.encoder.add_module(
                f'conv {i}',
                nn.Conv1d(self.conv_in_channels[i],
                          self.conv_in_channels[i + 1], self.conv_kernels[i],
                          self.conv_strides[i]))
            self.encoder.add_module(
                f'BN {i}', nn.BatchNorm1d(self.conv_in_channels[i + 1]))
            self.encoder.add_module(f'ac {i}', nn.LeakyReLU(0.2, inplace=True))
        temp = self.encoder(temp).size()
        self.mlp = nn.Linear(self.conv_in_channels[-1] * temp[-1],
                             self.hidden_size)

        self.decoder_input_size = self.embedding_size + self.hidden_size

        self.decoder = nn.LSTM(input_size=self.decoder_input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layer,
                               dropout=dropout)
        self.mlp_decoder = nn.Linear(self.hidden_size, self.vocab_size)

    def encode(self, X: torch.Tensor, is_noise=False):
        embeddings = self.embedding_layer(
            X)  # [batch, seq_len, embedding_size]
        embeddings = embeddings.permute(0, 2, 1)
        temp = self.encoder(embeddings)
        temp = temp.view(X.size()[0], -1)
        hidden = self.mlp(temp)
        # hidden [batch, hidden_size]
        # norms = torch.norm(hidden, p=2, dim=1)
        # hidden /= norms.unsqueeze(1)
        if is_noise:
            gaussian_noise = torch.normal(mean=torch.zeros_like(hidden),
                                          std=self.noise_std).to(
                                              Config.train_device)
            hidden += gaussian_noise
        return hidden

    def decode(self, X: torch.Tensor, hidden: torch.Tensor):
        all_hidden = hidden.unsqueeze(1)
        all_hidden = all_hidden.repeat(1, self.seq_len,
                                       1)  # [batch, seqlen, #]
        embeddings = self.embedding_decoder_layer(X)
        augmented_embeddings = torch.cat([embeddings, all_hidden], dim=2)
        outputs, _ = self.decoder(augmented_embeddings.permute(1, 0, 2))
        outputs = self.mlp_decoder(outputs).permute(
            1, 0, 2)  # [batch, seqlen, vocab_size]
        return outputs

    def forward(self,
                X,
                generator=None,
                inverter=None,
                encode_only=False,
                is_noise=False):
        hidden = self.encode(X, is_noise)
        if encode_only:
            return hidden
        if not generator:
            pass
            decoded = self.decode(X, hidden)
        else:
            z_hat = inverter(hidden)
            c_hat = generator(z_hat)
            decoded = self.decode(X, c_hat)
        return decoded


class MLP_D(nn.Module):
    def __init__(self, input_size, output_size, mlp_layer_sizes=[300, 300]):
        super(MLP_D, self).__init__()
        mlp_layer_sizes = [input_size] + mlp_layer_sizes
        self.mlp = nn.Sequential()
        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp.add_module(
                f'mlp {i}',
                nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1]))
            if i != 0:
                self.mlp.add_module(
                    f'bn {i}',
                    nn.BatchNorm1d(mlp_layer_sizes[i + 1],
                                   eps=1e-5,
                                   momentum=0.1))
            self.mlp.add_module(f'ac {i}', nn.LeakyReLU(0.2, inplace=True))
        self.mlp.add_module('mlp output',
                            nn.Linear(mlp_layer_sizes[-1], output_size))

    def forward(self, X):
        logits = self.mlp(X)
        return torch.mean(logits, dim=-1)


class MLP_G(nn.Module):
    def __init__(self, input_size, output_size, mlp_layer_sizes=[300, 300]):
        super(MLP_G, self).__init__()
        mlp_layer_sizes = [input_size] + mlp_layer_sizes
        self.mlp = nn.Sequential()
        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp.add_module(
                f'mlp {i}',
                nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1]))
            self.mlp.add_module(
                f'bn {i}',
                nn.BatchNorm1d(mlp_layer_sizes[i + 1], eps=1e-5, momentum=0.1))
            self.mlp.add_module(f'ac {i}', nn.ReLU(inplace=True))
        self.mlp.add_module('mlp output',
                            nn.Linear(mlp_layer_sizes[-1], output_size))

    def forward(self, X):
        logits = self.mlp(X)
        return logits


class MLP_I(nn.Module):
    def __init__(self, input_size, output_size, mlp_layer_sizes=[300, 300]):
        super(MLP_I, self).__init__()
        mlp_layer_sizes = [input_size] + mlp_layer_sizes
        self.mlp = nn.Sequential()
        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp.add_module(
                f'mlp {i}',
                nn.Linear(mlp_layer_sizes[i], mlp_layer_sizes[i + 1]))
            self.mlp.add_module(
                f'bn {i}',
                nn.BatchNorm1d(mlp_layer_sizes[i + 1], eps=1e-5, momentum=0.1))
            self.mlp.add_module(f'ac {i}', nn.ReLU(inplace=True))
        self.mlp.add_module('mlp output',
                            nn.Linear(mlp_layer_sizes[-1], output_size))

    def forward(self, X):
        logits = self.mlp(X)
        return logits


class JSDistance(nn.Module):
    def __init__(self, mean=0, std=1, epsilon=1e-5):
        super(JSDistance, self).__init__()
        self.epsilon = epsilon
        self.distrib_type_normal = True

    def get_kl_div(self, input, target):
        src_mu = torch.mean(input)
        src_std = torch.std(input)
        tgt_mu = torch.mean(target)
        tgt_std = torch.std(target)
        kl = torch.log(
            tgt_std / src_std) - 0.5 + (src_std**2 +
                                        (src_mu - tgt_mu)**2) / (2 *
                                                                 (tgt_std**2))
        return kl

    def forward(self, input, target):
        # KL(p, q) = log(sig2/sig1) + ((sig1^2 + (mu1 - mu2)^2)/2*sig2^2) - 1/2
        if self.distrib_type_normal:
            d1 = self.get_kl_div(input, target)
            d2 = self.get_kl_div(target, input)
            return 0.5 * (d1 + d2)
        else:
            input_num_zero = input.data[torch.eq(input.data, 0)]
            if input_num_zero.dim() > 0:
                input_num_zero = input_num_zero.size(0)
                input.data = input.data - (self.epsilon / input_num_zero)
                input.data[torch.lt(input.data,
                                    0)] = self.epsilon / input_num_zero
            target_num_zero = target.data[torch.eq(target.data, 0)]
            if target_num_zero.dim() > 0:
                target_num_zero = target_num_zero.size(0)
                target.data = target.data - (self.epsilon / target_num_zero)
                target.data[torch.lt(target.data,
                                     0)] = self.epsilon / target_num_zero
            d1 = torch.sum(input * torch.log(input / target)) / input.size(0)
            d2 = torch.sum(target * torch.log(target / input)) / input.size(0)
            return (d1 + d2) / 2


if __name__ == "__main__":
    train_data = 'Fears for T N pension after talks. Unions representing workers at Turner   Newall say they are \'disappointed\' after talks with stricken parent firm Federal Mogul.'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data_tokens = (tokenizer.tokenize(train_data + ' [SEP]'))
    label_tokens = (tokenizer.tokenize('[CLS] ' + train_data))
    data_idx = []
    label_idx = []
    data_mask = []
    data_idx.append(tokenizer.convert_tokens_to_ids(data_tokens))
    data_mask.append([1] * len(data_tokens))
    label_idx.append(tokenizer.convert_tokens_to_ids(label_tokens))
    data_idx = torch.tensor(data_idx)
    data_mask = torch.tensor(data_mask)
    label_idx = torch.tensor(label_idx)
    Seq2Seq_model_bert = Seq2Seq_bert(embedding_size=Config.embedding_size,
                                      hidden_size=Config.hidden_size).to(
                                          Config.train_device)
    criterion_baseline_model = nn.CrossEntropyLoss().to(Config.train_device)
    optimizer_baseline_model = optim.Adam(Seq2Seq_model_bert.parameters(),
                                          lr=Config.baseline_train_rate)
    data_idx = data_idx.to(Config.train_device)
    data_mask = data_mask.to(Config.train_device)
    label_idx = label_idx.to(Config.train_device)
    logits = Seq2Seq_model_bert(data_idx, data_mask, is_noise=True)
    optimizer_baseline_model.zero_grad()
    loss = criterion_baseline_model(logits, label_idx)
    loss.backward()
    optimizer_baseline_model.step()
    outputs_idx = logits.arguemax(dim=2)
    outputs_tokens = tokenizer.convert_ids_to_tokens(outputs_idx)
    print(outputs_tokens)
    