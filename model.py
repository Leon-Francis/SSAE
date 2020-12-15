import torch
import torch.nn as nn
from config import Config
from transformers import BertModel, BertConfig


class Seq2Seq_bert(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.2,
                 noise_std=0.2):
        super(Seq2Seq_bert, self).__init__()
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
        self.decoder = nn.LSTM(input_size=768,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout=self.dropout)

    def forward(self, inputs, inputs_mask, is_noise=False):
        encoder, pooled = self.encoder(inputs, attention_mask=inputs_mask)


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


if __name__ == '__main__':

    # stride * (i-1) + kernel  = X
    # torch.set_printoptions(threshold=5000)

    X = torch.randint(low=0, high=10, size=[16, 20])
    X = X.to(Config.train_device)

    # print(X)
    net = Seq2SeqAE(10, 20, 100, 10).to(Config.train_device)
    gan_disc = MLP_D(10, 1)
