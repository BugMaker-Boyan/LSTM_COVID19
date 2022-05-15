import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, args, device=None):
        super(LSTM, self).__init__()
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.device = device
        self.lstm = nn.LSTM(args.input_dim, args.hidden_dim, args.num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device, dtype=torch.float)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device, dtype=torch.float)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

