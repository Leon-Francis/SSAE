import torch
import torch.nn as nn
import heapq
from config import AttackConfig
from baseline_module.baseline_model_builder import BaselineModelBuilder


class Node(nn.Module):
    def __init__(self, token, freq):
        super(Node, self).__init__()
        self.linear = nn.Linear(AttackConfig.hidden_size,
                                2).to(AttackConfig.train_device)
        self.softmax = nn.Softmax(dim=1).to(AttackConfig.train_device)
        self.token = token
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

    def __gt__(self, other):
        return self.freq > other.freq

    def __eq__(self, other):
        if (other is None):
            return False
        if (not isinstance(other, Node)):
            return False
        return self.freq == other.freq

    def __hash__(self):
        return hash(self.token)

    def forward(self, hidden):
        hidden = hidden.unsqueeze(dim=0)
        return self.softmax(self.linear(hidden)).squeeze(dim=0)


class HuffmanTree(nn.Module):
    def __init__(self, word_count, attack_vocab):
        super(HuffmanTree, self).__init__()
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}
        self.vocab = attack_vocab
        self.root = None
        self.criterion_ce = nn.CrossEntropyLoss()
        self.make_heap(word_count, attack_vocab)
        self.merge_nodes()
        self.make_codes()

    def make_heap(self, frequency, vocab):
        for key in frequency:
            node = Node(vocab.get_index(key), frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while (len(self.heap) > 1):
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = Node(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if (root is None):
            return
        if (root.token is not None):
            self.codes[root.token] = current_code
            self.reverse_mapping[current_code] = root.token
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.root = root
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_word(self, code):
        return self.reverse_mapping.get(code)

    def get_code(self, word):
        if self.codes.get(word.item()) is None:
            return self.codes['<unk>']  # unknown word
        return self.codes[word.item()]

    def forward(self, hidden, target):
        """return loss

        Args:
            hidden (tensor): [batch, sen_len, hidden_size]
            target (tensor): [batch, sen_len]

        Returns:
            [tensor]: [1]
        """
        # hidden [batch * seq_len, hidden_size]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        # target [batch * seq_len]
        target = target.reshape(-1)
        loss = None
        for i in range(hidden.shape[0]):
            path_to_word = self.get_code(target[i])
            root = self.root
            for choice in path_to_word:
                if loss is None:
                    loss = self.criterion_ce(
                        root(hidden[i]).unsqueeze(dim=0),
                        torch.tensor([int(choice)
                                      ]).to(AttackConfig.train_device))
                else:
                    loss += self.criterion_ce(
                        root(hidden[i]).unsqueeze(dim=0),
                        torch.tensor([int(choice)
                                      ]).to(AttackConfig.train_device))
                if (choice == '0'):
                    root = root.left
                else:
                    root = root.right

        return loss

    def get_sentence(self, hidden):
        """get sentence from hidden

        Args:
            hidden (tensor): [batch, sen_len, hidden_size]

        Returns:
            [tensor]: [batch, sen_len]
        """
        sen_len = hidden.shape[1]
        # hidden [batch * seq_len, hidden_size]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        sentence = []
        for i in range(hidden.shape[0]):
            root = self.root
            while True:
                choice = root(hidden[i]).argmax(dim=0).item()
                if root.token is not None:
                    sentence.append(root.token)
                    break
                if (choice == 0):
                    root = root.left
                else:
                    root = root.right
        return torch.tensor(sentence).view(-1, sen_len).to(
            AttackConfig.train_device)


if __name__ == '__main__':

    baseline_model_builder = BaselineModelBuilder('AGNEWS',
                                                  'LSTM',
                                                  AttackConfig.train_device,
                                                  is_load=True)
    word_count = {
        k: v
        for k, v in sorted(baseline_model_builder.vocab.word_count.items(),
                           key=lambda x: x[1],
                           reverse=True)
    }
    ht = HuffmanTree(word_count, baseline_model_builder.vocab)
    pass
