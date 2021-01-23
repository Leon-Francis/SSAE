import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import nn, optim


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
    import argparse
    from transformers import BertTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='AGNEWS')
    parser.add_argument('--model', default='Bert')
    args = parser.parse_args()


    text = 'hello, i love u ..\n'

    tokenzier = BertTokenizer.from_pretrained('bert-base-uncased')

    x = tokenzier(text, truncation=True, max_length=10, padding=True)
    print(x)
    # device = torch.device('cuda:3')
    # model = BaselineModelBuilder(args.dataset, args.model, device, is_load=False)
    # x = torch.tensor([0, 1, 2, 0], dtype=torch.long).to(device)
    # types = torch.tensor([0, 0, 0, 0], dtype=torch.long).to(device)
    # masks = torch.tensor([1, 1, 1, 1], dtype=torch.long).to(device)
    # predicts = model.predict_class(x, types, masks)
    #
    # print(predicts)



