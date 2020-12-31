from torch import nn
from torch.nn import init
from transformers import BertModel, BertConfig
from config import Config


class Baseline_Model_Bert(nn.Module):
    def __init__(self):
        super(Baseline_Model_Bert, self).__init__()
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.linear = nn.Linear(768, Config.label_num)
        for params in self.linear.parameters():
            init.normal_(params, mean=0, std=0.01)

    def forward(self, inputs, inputs_mask):
        """forward

        Args:
            inputs (torch.tensor): [batch, seq_len]
            inputs_mask (torch.tensor): [batch, seq_len]

        Returns:
            logits: [batch, 4]
        """
        encoder, pooled = self.bert_model(inputs, attention_mask=inputs_mask)
        logits = self.linear(pooled)
        return logits
