from datetime import datetime
from config import Config
import csv


def logging(info: str):
    print('\n\r' + '[INFO]' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          '\n\r' + str(info))


def read_AGNEWS_origin_data(data_path):
    datas = []
    labels = []
    with open(data_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            labels.append(int(line[0]) - 1)
            datas.append(line[1] + '. ' + line[2])
    return datas, labels


def read_standard_data(path):
    data = []
    labels = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            data.append(line[:-1])
            labels.append(int(line[-1]))
    logging(f'loading data {len(data)} from {path}')
    return data, labels


def write_standard_data(datas, labels, path, mod='w'):
    assert len(datas) == len(labels)
    num = len(labels)
    logging(f'writing standard data {num} to {path}')
    with open(path, mod, newline='', encoding='utf-8') as file:
        for i in range(num):
            file.write(datas[i] + str(labels[i]) + '\n')


def read_standard_data4Test(path):
    data = []
    labels = []
    i = 320
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            i -= 1
            line = line.strip('\n')
            data.append(line[:-1])
            labels.append(int(line[-1]))
            if i == 0:
                break
    logging(f'loading data {len(data)} from {path}')
    return data, labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=Config.dataset_list)
    parser.add_argument('--path', default=None)
    parser.add_argument('--output_path', default=None)
    args = parser.parse_args()

    dataset = args.dataset
    path = args.path
    output_path = args.output_path
    if dataset == 'AGNEWS':
        datas, labels = read_AGNEWS_origin_data(path)
        write_standard_data(datas, labels, output_path)
