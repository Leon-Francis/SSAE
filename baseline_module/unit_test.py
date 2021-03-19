import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import nn, optim
from baseline_config import setup_seed

setup_seed(667)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(10, 5)
        self.gru = nn.GRU(5, 6)

    def forward(self, x):
        x = x.permute(1, 0)
        t = self.embedding(x)
        outputs, hiddens = self.gru(t)

        return outputs, hiddens

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(20, 5)
        self.gru = nn.GRU(5, 6)
        self.out = nn.Linear(6, 20)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.permute(1, 0)
        input = self.embedding(input)
        input = F.relu(input)
        output, hidden = self.gru(input, hidden)
        # output = self.softmax(self.out(output))
        return self.out(output), hidden

def testSeq2Seq():
    x = torch.tensor([[3, 9, 4, 1]], dtype=torch.long)
    encoder = Encoder()
    decoder = Decoder()
    xx = torch.tensor([[0, 2, 6, 19]], dtype=torch.long)
    y = torch.tensor([[2, 6, 19, 1]], dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ], lr=1e-3
    )

    for epoch in range(100):
        outputs, hiddens = encoder(x)
        outputs, hiddens = decoder(xx, hiddens)
        loss = criterion(outputs.view([-1, 20]), y.view([-1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss', loss.item())
        outputs = torch.log_softmax(outputs, dim=-1)
        predicts = outputs.argmax(dim=-1)
        print('outputs', predicts)

if __name__ == '__main__':
    # import argparse
    # from transformers import BertTokenizer
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='AGNEWS')
    # parser.add_argument('--model', default='Bert')
    # args = parser.parse_args()


    # from baseline_config import *
    # from baseline_data import baseline_MyDataset
    # from baseline_vocab import baseline_Vocab
    # from torch.utils.data import DataLoader
    # from baseline_nets import baseline_LSTM_Entailment, baseline_TextCNN_Entailment
    # dataset_config = baseline_config_dataset['SNLI']
    #
    # train_dataset = baseline_MyDataset('SNLI', dataset_config.train_data_path)
    # vocab = baseline_Vocab(train_dataset.data_token['pre']+train_dataset.data_token['hypo'],
    #                               is_using_pretrained=True, is_special=True,
    #                               vocab_limit_size=dataset_config.vocab_limit_size,
    #                               word_vec_file_path=dataset_config.pretrained_word_vectors_path)
    # train_dataset.token2seq(vocab, maxlen=dataset_config.padding_maxlen)
    #
    # print(len(train_dataset))
    # # print(train_dataset[0])
    #
    # # net = baseline_LSTM_Entailment(50, 1, 100, vocab, 3, using_pretrained=True, bid=True, head_tail=False)
    # net = baseline_TextCNN_Entailment(vocab, 50, is_static=False, using_pretrained=True, num_channels=[50, 50, 50],
    #                                   kernel_sizes=[3, 4, 5], labels_num=3, is_batch_normal=True)
    # train_dataset = DataLoader(train_dataset, batch_size=64, shuffle=True)
    #
    #
    # for (x_pre, x_hypo, label) in train_dataset:
    #     pass
        # print(x_pre)
        # print(x_hypo)
        # print(label)

    # import torch
    # from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
    #
    # # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
    # import logging
    #
    # logging.basicConfig(level=logging.INFO)
    #
    # # Load pre-trained model tokenizer (vocabulary)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # text = '[CLS] I want to [MASK] the car because it is cheap . [SEP]'
    # tokenized_text = tokenizer.tokenize(text)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #
    # # Create the segments tensors.
    # segments_ids = [0] * len(tokenized_text)
    # # Convert inputs to PyTorch tensors
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    #
    # # Load pre-trained model (weights)
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    # model.eval()
    # masked_index = tokenized_text.index('[MASK]')
    #
    # # Predict all tokens
    # with torch.no_grad():
    #     predictions = model(tokens_tensor, segments_tensors)
    #
    # predicted_index = torch.argmax(predictions[0, masked_index]).item()
    # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    #
    # print(predicted_token)

    import torch
    from torch import nn

    max_len = 30
    word_dim = 100
    vocab_size = 100
    batch_size = 64
    hidden_dim = 100

    embedding = nn.Embedding(vocab_size, word_dim)
    net = nn.LSTM(word_dim, hidden_dim)
    sentence = torch.randint(0, vocab_size, [max_len, batch_size])
    sentence_len = torch.randint(1, max_len+1, [batch_size])

    print(sentence)

    print(sentence_len)

    def packed():
        slen_sort, idx_sort = torch.sort(sentence_len, descending=True)
        idx_reverse = torch.argsort(idx_sort)

        s = sentence.index_select(1, idx_sort)
        s_packed = nn.utils.rnn.pack_padded_sequence(s, slen_sort)
        outs, _ = net(s_packed)

        temp, _ = nn.utils.rnn.pad_packed_sequence(outs, padding_value=0)

        temp = temp.index_select(1, idx_reverse)

        temp = temp[sentence_len-1, torch.arange(temp.size()[1]), :]

        return temp

    def normal():

        outs, _ = net(sentence)

        temp = outs[sentence_len-1, torch.arange(outs.size()[1]), :]

        return temp


    sentence = embedding(sentence)

    t1 = packed()

    t2 = normal()

    print(t1)

    print(t2)

    print(torch.eq(t1, t2))

    for i in range(0):
        print(i)






    pass
