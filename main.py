import os.path
import numpy as np
import torch.nn
from arguments import parser
from dataset.COVID_dataset import COVIDDataset
from model.LSTM_model import LSTM
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


args = parser.parse_args()

device = torch.device('cpu')
if args.use_gpu and torch.cuda.is_available():
    device = torch.device('cuda')

covid_train_dataset = COVIDDataset(args=args, test=False)
covid_test_dataset = COVIDDataset(args=args, test=True)

summary_writer = SummaryWriter(log_dir=args.log_dir)

model = LSTM(args=args, device=device).to(device)
model.train()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

train_loss_list = []
test_loss_list = []
best_loss = np.Inf

for epoch in range(args.num_epochs):
    train_data_loader = DataLoader(dataset=covid_train_dataset, batch_size=args.batch_size, shuffle=True)
    for idx, (x, y) in enumerate(train_data_loader):
        x, y = x.to(device), y.to(device)
        y_pre = model(x)
        loss = criterion(y_pre, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    test_data_loader = DataLoader(dataset=covid_test_dataset, batch_size=args.batch_size, shuffle=False)
    for idx, (x, y) in enumerate(test_data_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pre = model(x)
            loss = criterion(y_pre, y)

        test_loss_list.append(loss.item())

    summary_writer.add_scalars('loss',
                               {'train_loss': np.mean(train_loss_list), 'test_loss': np.mean(test_loss_list)}, epoch)

    print(f'Epoch: [{epoch}/{args.num_epochs}], '
          f'Train Loss: {np.mean(train_loss_list)}, '
          f'Test Loss: {np.mean(test_loss_list)}')

    if np.mean(test_loss_list) < best_loss:
        save_path = os.path.join(args.checkpoints, args.data_name + '_' + args.target + '.pth')
        torch.save(model.state_dict(), save_path)
