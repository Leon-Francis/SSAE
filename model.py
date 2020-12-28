import torch
import torch.nn as nn
from config import Config
from transformers import BertModel

from transformers import BertTokenizer
from torch import optim


class Seq2Seq_bert(nn.Module):
    def __init__(self, hidden_size, num_layers=1, dropout=0.0, noise_std=0.2):
        super(Seq2Seq_bert, self).__init__()
        self.seq_len = Config.sen_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_std = noise_std
        self.encoder = BertModel.from_pretrained('bert-base-uncased',
                                                 output_hidden_states=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
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
        encoders, pooled, all_hidden_states = self.encoder(
            inputs, attention_mask=inputs_mask)
        # pooled [batch, hidden_size]
        # hidden [batch, seq_len, hidden_size]
        hidden = encoders
        state = all_hidden_states[0]  # embedding
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
    Seq2Seq_model_bert = Seq2Seq_bert(hidden_size=Config.hidden_size).to(
        Config.train_device)
    criterion_baseline_model = nn.CrossEntropyLoss().to(Config.train_device)
    optimizer_baseline_model = optim.Adam(Seq2Seq_model_bert.parameters(),
                                          lr=Config.baseline_learning_rate)
    data_idx = data_idx.to(Config.train_device)
    data_mask = data_mask.to(Config.train_device)
    label_idx = label_idx.to(Config.train_device)
    logits = Seq2Seq_model_bert(data_idx, data_mask, is_noise=True)
    optimizer_baseline_model.zero_grad()
    logits_flatten = logits.view(-1, logits.shape[-1])
    label_idx_flatten = label_idx.view(-1)
    loss = criterion_baseline_model(logits_flatten, label_idx_flatten)
    loss.backward()
    optimizer_baseline_model.step()
    outputs_idx = logits.argmax(dim=2)
    outputs_tokens = []
    for i in range(outputs_idx.shape[0]):
        outputs_tokens.append(tokenizer.convert_ids_to_tokens(outputs_idx[i]))
    for tokens in outputs_tokens:
        print(' '.join(tokens))
    print(' '.join(label_tokens))
    print(f'loss is {loss}')
