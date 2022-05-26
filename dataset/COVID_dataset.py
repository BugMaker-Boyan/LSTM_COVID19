import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class COVIDDataset(Dataset):
    def __init__(self, args, test=False):
        super(COVIDDataset, self).__init__()
        _data = pd.read_csv(args.data_path)
        _data = _data.sort_values(args.date_column)
        _data = _data[args.target].values
        self.args = args

        self.data = []
        for index in range(len(_data) - args.seq_len):
            self.data.append(_data[index:index + args.seq_len])
        self.data = np.array(self.data)

        _test_set_size = int(np.round(args.test_size * self.data.shape[0]))
        _train_set_size = self.data.shape[0] - _test_set_size

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(_data[:_train_set_size + args.seq_len - 2].reshape(-1, 1))

        if not test:
            self.x = self.data[:_train_set_size, :-1]
            self.y = self.data[:_train_set_size, -1]
        else:
            self.x = self.data[_train_set_size:, :-1]
            self.y = self.data[_train_set_size:, -1]

    def __getitem__(self, item):
        x = self.scaler.transform(self.x[item, :].reshape(-1, 1))
        if self.args.inverse:
            y = np.expand_dims(np.array(self.y[item]), axis=0)
        else:
            y = self.scaler.transform(np.expand_dims(np.array(self.y[item]), axis=0).reshape(-1, 1)).squeeze(axis=0)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def __len__(self):
        return self.y.shape[0]

    def inverse(self, x):
        return self.scaler.inverse_transform(x)
