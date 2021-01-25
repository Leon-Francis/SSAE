import sys
sys.path.append('./baseline_module')
import torch
import baseline_module


device = torch.device('cpu')
# net = baseline_module.baseline_model_builder.BaselineModelBuilder('AGNEWS', 'TextCNN', device)
# x = torch.tensor(torch.randint(low=0, high=1000, size=[2, 300]), dtype=torch.long).to(device)
#
# net.set_eval_mode()
# print(net(x))
# print(net.predict_class(x))
from baseline_model import Baseline_Model_LSTM_Entailment
from config import SNLIConfig
from data import SNLI_Dataset

# net = Baseline_Model_LSTM_Entailment(SNLIConfig, False)
dataset = SNLI_Dataset(train_data=True, debug_mode=True)

print(dataset[0][-4:])
# x = torch.randint(low=0, high=3000, size=[2, 5, 10])
# x, y = x[0], x[1]
#
# ans = net.forward(x, y)
# print(ans)

