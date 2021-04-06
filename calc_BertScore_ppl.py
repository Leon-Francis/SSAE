from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
import torch
from collections import defaultdict
from bert_score.utils import lang2model, model2layers, bert_cos_score_idf
from transformers import AutoModel, AutoTokenizer
import numpy as np
import math


def pre_bertscore(device):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model_type = lang2model['en']
    num_layers = model2layers[model_type]
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.encoder.layer = \
        torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

    # not IDF
    idf_dict = defaultdict(lambda: 1.)
    # set idf for [SEP] and [CLS] to 0
    idf_dict[tokenizer.sep_token_id] = 0
    idf_dict[tokenizer.cls_token_id] = 0
    idf_dict[tokenizer.unk_token_id] = 0
    idf_dict[0] = 0

    model.to(device)
    model.eval()

    return model, tokenizer, idf_dict


def get_bertscore(model, tokenizer, idf_dict, refs, cands, device):
    with torch.no_grad():
        all_preds = bert_cos_score_idf(model,
                                       refs,
                                       cands,
                                       tokenizer,
                                       idf_dict,
                                       verbose=False,
                                       device=device,
                                       batch_size=64,
                                       all_layers=None)
    P = all_preds[..., 0].cpu()
    R = all_preds[..., 1].cpu()
    F1 = all_preds[..., 2].cpu()
    return P, R, F1


def pre_ppl(device):
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    return enc, model


def get_ppl(enc, model, cands, device):
    ppls = []
    with torch.no_grad():
        for s in cands:  # actually here is a batch with batchsize=1
            try:
                s = enc.encode(s) + [
                    50256
                ]  # 50256 is the token_id for <|endoftext|>
                batch = torch.tensor([s]).to(device)

                loss = model(batch, lm_labels=batch)  # everage -logp

                ppls.append(math.exp(loss.item()))  # the small, the better
            except KeyError:
                continue
    return ppls


def calc_bert_score_ppl(cands_dir, refs_dir, device):
    with open(cands_dir, encoding='utf8') as f:
        cands = [line.strip() for line in f]
    with open(refs_dir, encoding='utf8') as f:
        refs = [line.strip() for line in f]
    ppl_enc, ppl_model = pre_ppl(device)
    ppl = get_ppl(ppl_enc, ppl_model, cands, device)
    bs_model, bs_tokenizer, bs_idf_dict = pre_bertscore(device)
    P, R, F = get_bertscore(bs_model, bs_tokenizer, bs_idf_dict, refs, cands,
                            device)
    return np.mean(ppl), F.mean().item()


if __name__ == '__main__':
    with open('./texts/PWWS/AGNEWS/LSTM/cands.txt', encoding='utf8') as f:
        cands = [line.strip()[:-1] for line in f]
    with open('./texts/PWWS/AGNEWS/LSTM/refs.txt', encoding='utf8') as f:
        refs = [line.strip()[:-1] for line in f]
    ppl_enc, ppl_model = pre_ppl()
    ppl = get_ppl(ppl_enc, ppl_model, refs)
    print(np.mean(ppl))
    bs_model, bs_tokenizer, bs_idf_dict = pre_bertscore()
    P, R, F = get_bertscore(bs_model, bs_tokenizer, bs_idf_dict, refs, cands)
    print(F.mean().item())
