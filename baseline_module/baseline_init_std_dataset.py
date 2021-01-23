import argparse

from baseline_tools import write_standard_data, read_IMDB_origin_data, read_AGNEWS_origin_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--path')
parser.add_argument('--output')
args = parser.parse_args()

dataset = args.dataset
path = args.path
output = args.output


if dataset == 'IMBD':
    datas, labels = read_IMDB_origin_data(path)
    write_standard_data(datas, labels, output)
elif dataset == 'AGNEWS':
    datas, labels = read_AGNEWS_origin_data(path)
    write_standard_data(datas, labels, output)
else:
    raise NotImplementedError()
