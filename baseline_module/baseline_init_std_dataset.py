import argparse
import pandas as pd

from baseline_tools import write_standard_data, read_IMDB_origin_data, read_AGNEWS_origin_data, \
    read_SST2_origin_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--path')
parser.add_argument('--output')
args = parser.parse_args()

dataset = args.dataset
path = args.path
output = args.output


def create_SST2(data_name, sst_folder, output_folder):
    if sst_folder[-1] != '/':
        sst_folder += '/'
    if output_folder[-1] != '/':
        output_folder += '/'
    datasetSentences = pd.read_csv(sst_folder + 'datasetSentences.txt', sep='\t')
    dictionary = pd.read_csv(sst_folder + 'dictionary.txt', sep='|', header=None, names=['sentence', 'phrase ids'])
    datasetSplit = pd.read_csv(sst_folder + 'datasetSplit.txt', sep=',')
    sentiment_labels = pd.read_csv(sst_folder + 'sentiment_labels.txt', sep='|')

    # 将多个表进行内连接合并
    dataset = pd.merge(pd.merge(pd.merge(datasetSentences, datasetSplit), dictionary), sentiment_labels)

    def labeling(data_name, sentiment_value):
        if sentiment_value <= 0.4:
            return 0  # negative
        elif sentiment_value > 0.6:
            return 1  # positive
        else:
            return -1  # drop neutral

    def check_not_punctuation(token):
        for ch in token:
            if ch.isalnum(): return True
        return False

    def filter_punctuation(s):
        s = s.lower().split(' ')
        return ' '.join([token for token in s if check_not_punctuation(token)])

    dataset['sentiment_label'] = dataset['sentiment values'].apply(lambda x: labeling(data_name, x))
    dataset = dataset[dataset['sentiment_label'] != -1]
    dataset['sentence'] = dataset['sentence'].apply(lambda s: filter_punctuation(s))


    # 保存处理好的数据集
    # train
    dataset[dataset['splitset_label'] == 1][['sentence', 'sentiment_label']].to_csv(
        output_folder + data_name + '_' + 'train.csv', index=False)
    # test
    dataset[dataset['splitset_label'] == 2][['sentence', 'sentiment_label']].to_csv(
        output_folder + data_name + '_' + 'test.csv', index=False)
    # dev
    dataset[dataset['splitset_label'] == 3][['sentence', 'sentiment_label']].to_csv(
        output_folder + data_name + '_' + 'dev.csv', index=False)

    train_datas, train_labels = read_SST2_origin_data(output_folder + data_name + '_' + 'train.csv')
    dev_datas, dev_labels = read_SST2_origin_data(output_folder + data_name + '_' + 'dev.csv')
    test_data, test_labels = read_SST2_origin_data(output_folder + data_name + '_' + 'test.csv')

    write_standard_data(train_datas+dev_datas, train_labels+dev_labels, path=output_folder+'train.std')
    write_standard_data(test_data, test_labels, path=output_folder+'test.std')



if dataset == 'IMBD':
    datas, labels = read_IMDB_origin_data(path)
    write_standard_data(datas, labels, output)
elif dataset == 'AGNEWS':
    datas, labels = read_AGNEWS_origin_data(path)
    write_standard_data(datas, labels, output)
elif dataset == 'SST2':
    create_SST2(dataset, path, output)
else:
    raise NotImplementedError()
