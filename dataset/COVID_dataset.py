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
        _data = _data[args.target].values.reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        _data = self.scaler.fit_transform(_data).squeeze()

        self.data = []
        for index in range(len(_data) - args.seq_len):
            self.data.append(_data[index:index + args.seq_len])
        self.data = np.array(self.data)

        _test_set_size = int(np.round(args.test_size * self.data.shape[0]))
        _train_set_size = self.data.shape[0] - _test_set_size

        if not test:
            self.x = self.data[:_train_set_size, :-1, np.newaxis]
            self.y = self.data[:_train_set_size, -1, np.newaxis]
        else:
            self.x = self.data[_train_set_size:, :-1, np.newaxis]
            self.y = self.data[_train_set_size:, -1, np.newaxis]

    def __getitem__(self, item):
        return torch.tensor(self.x[item, :, :], dtype=torch.float), \
               torch.tensor(self.y[item, :], dtype=torch.float)

    def __len__(self):
        return self.y.shape[0]

