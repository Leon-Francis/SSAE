import torch


class Config:
    """configuration
    """

    train_device = torch.device('cuda:1')
    dataset_list = ['IMDB', 'AGNEWS', 'YAHOO']
    label_num = 4
    train_data_path = r'./dataset/AGNEWS/train.std'
    test_data_path = r'./dataset/AGNEWS/test.std'
    output_dir = r'./output'

    epochs = 20
    batch_size = 128

    sen_size = 70


if __name__ == "__main__":
    print(Config.config_device)
