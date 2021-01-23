import csv
import os
import pickle
from datetime import datetime

if os.path.exists('./baseline_module'):
    os.chdir('./baseline_module')

# print(os.getcwd())

def load_pkl_obj(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    return None

def save_pkl_obj(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def logging(info: str):
    print('\n\r' + '[INFO]' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
          '\n\r' + str(info))

def get_time()->str:
    return str(datetime.now().strftime("%m_%d-%H-%M"))

def parse_bool(v):
    return 'y' in v

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
            file.write(datas[i]+str(labels[i])+'\n')

def read_IMDB_origin_data(data_path):
    path_list = []
    logging(f'start loading data from {data_path}')
    dirs = os.listdir(data_path)
    for dir in dirs:
        if dir == 'pos' or dir == 'neg':
            file_list = os.listdir(os.path.join(data_path, dir))
            file_list = map(lambda x: os.path.join(data_path, dir, x), file_list)
            path_list += list(file_list)
    datas = []
    labels = []
    for p in path_list:
        label = 0 if 'neg' in p else 1
        with open(p, 'r', encoding='utf-8') as file:
            datas.append(file.readline())
            labels.append(label)

    return datas, labels

def read_AGNEWS_origin_data(data_path):
    datas = []
    labels = []
    with open(data_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
            labels.append(int(line[0]) - 1)
            datas.append(line[1] + '. ' + line[2])
    return datas, labels

if __name__ == '__main__':
    pass