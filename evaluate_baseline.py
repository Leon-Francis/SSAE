import torch
from torch.utils.data import DataLoader
from baseline_module.baseline_model_builder import BaselineModelBuilder
from config import BaselineConfig
from data import AGNEWS_Dataset
from tools import logging


def eval_bert_baseline_Classification(model, test_data):
    with torch.no_grad():
        model.eval()
        acc_sum = 0
        n = 0
        for _, x_mask, x, label in test_data:
            x, x_mask, label = x.to(BaselineConfig.train_device), x_mask.to(
                BaselineConfig.train_device), label.to(
                    BaselineConfig.train_device)
            logits = model(x, x_mask)
            acc_sum += (logits.argmax(dim=1) == label).float().sum().item()
            n += label.shape[0]
        return acc_sum / n


if __name__ == "__main__":
    baseline_model_builder = BaselineModelBuilder('AGNEWS',
                                                  'LSTM',
                                                  BaselineConfig.train_device,
                                                  is_load=True)
    test_dataset_orig = AGNEWS_Dataset(
        train_data=False,
        attack_vocab=baseline_model_builder.vocab,
        debug_mode=False)
    test_data = DataLoader(test_dataset_orig,
                           batch_size=BaselineConfig.batch_size,
                           shuffle=False,
                           num_workers=4)
    logging(
        eval_bert_baseline_Classification(baseline_model_builder.net,
                                          test_data))
