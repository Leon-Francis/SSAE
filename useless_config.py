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

    baseline_learning_rate = 1e-4
    Seq2Seq_learning_rate = 1e-4
    gan_gen_learning_rate = 1e-4
    gan_adv_learning_rate = 1e-5
    optim_betas = (0.9, 0.999)

    hidden_size = 768
    super_hidden_size = 500
    vocab_size = bert_vocab_size
    sen_len = 20
    # new

    load_pretrained_Seq2Seq = False
    fine_tuning = False
    gan_gen_train_model = False


config_model_load_path = {
    'IMDB': {
        'LSTM': r'./models/IMDB/LSTM.pt',
        'LSTM_enhanced': r'./models/IMDB/LSTM_enhanced.pt',
        'TextCNN': r'./models/IMDB/TextCNN.pt',
        'TextCNN_enhanced': r'./models/IMDB/TextCNN_enhanced.pt',
        'BidLSTM': r'./models/IMDB/BidLSTM.pt',
        'BidLSTM_enhanced': r'./models/IMDB/BidLSTM_enhanced.pt',
        'LSTM_adv': r'./models/IMDB/LSTM_adv.pt',
        'BidLSTM_adv': r'./models/IMDB/BidLSTM_adv.pt',
        'TextCNN_adv': r'./models/IMDB/TextCNN_adv.pt',
    },
    'AGNEWS': {
        'LSTM': r'./models/AGNEWS/LSTM.pt',
        'BidLSTM': r'./models/AGNEWS/BidLSTM.pt',
        'TextCNN': r'./models/AGNEWS/TextCNN.pt',
        'LSTM_enhanced': r'./models/AGNEWS/LSTM_enhanced.pt',
        'TextCNN_enhanced': r'./models/AGNEWS/TextCNN_enhanced.pt',
        'BidLSTM_enhanced': r'./models/AGNEWS/BidLSTM_enhanced.pt',
        'LSTM_adv': r'./models/AGNEWS/LSTM_adv.pt',
        'BidLSTM_adv': r'./models/AGNEWS/BidLSTM_adv.pt',
        'TextCNN_adv': r'./models/AGNEWS/TextCNN_adv.pt',
    },
    'YAHOO': {
        'TextCNN': r'./models/YAHOO/TextCNN.pt',
        'LSTM': r'./models/YAHOO/LSTM.pt',
        'BidLSTM': r'./models/YAHOO/BidLSTM.pt',
        'TextCNN_enhanced': r'./models/YAHOO/TextCNN_enhanced.pt',
        'LSTM_enhanced': r'./models/YAHOO/LSTM_enhanced.pt',
        'BidLSTM_enhanced': r'./models/YAHOO/BidLSTM_enhanced.pt',
        'LSTM_adv': r'./models/YAHOO/LSTM_adv.pt',
        'BidLSTM_adv': r'./models/YAHOO/BidLSTM_adv.pt',
        'TextCNN_adv': r'./models/YAHOO/TextCNN_adv.pt',
    }
}
