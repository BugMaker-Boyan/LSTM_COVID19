import matplotlib.pyplot as plt
import torch
from dataset.COVID_dataset import COVIDDataset
from torch.utils.data import DataLoader
from model.LSTM_model import LSTM
from arguments import parser
import os

args = parser.parse_args()
device = torch.device('cpu')
model_path = os.path.join(args.checkpoints, args.data_name + '_' + args.target + '.pth')

model = LSTM(args, device)

model.load_state_dict(torch.load(model_path, map_location='cpu'))

model.eval()

covid_train_dataset = COVIDDataset(args, test=False)
covid_test_dataset = COVIDDataset(args, test=True)

train_data_loader = DataLoader(dataset=covid_train_dataset, batch_size=1, shuffle=False)
test_data_loader = DataLoader(dataset=covid_test_dataset, batch_size=1, shuffle=False)

x_list = []
y_real_list = []
y_pre_list = []

for idx, (x, y) in enumerate(train_data_loader):
    with torch.no_grad():
        y_pre = model(x)
        x_list.append(idx)
        if args.inverse:
            y_real_list.append(covid_train_dataset.inverse(y).item())
            y_pre_list.append(covid_train_dataset.inverse(y_pre).item())
        else:
            y_real_list.append(y.item())
            y_pre_list.append(y_pre.item())

for idx, (x, y) in enumerate(test_data_loader):
    with torch.no_grad():
        y_pre = model(x)
        x_list.append(idx + len(train_data_loader))
        if args.inverse:
            y_real_list.append(covid_train_dataset.inverse(y).item())
            y_pre_list.append(covid_train_dataset.inverse(y_pre).item())
        else:
            y_real_list.append(y.item())
            y_pre_list.append(y_pre.item())

plt.plot(x_list, y_real_list, label='real', c='red')
plt.plot(x_list, y_pre_list, label='prediction', c='blue')

plt.vlines(len(train_data_loader), 0, 1, linestyles='dashed')

plt.legend()

plt.title(args.target)

plt.show()

