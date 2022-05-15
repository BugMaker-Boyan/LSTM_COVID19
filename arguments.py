import argparse

parser = argparse.ArgumentParser(description='COVID19 LSTM Prediction Model')

parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--date_column', type=str, default='date')
parser.add_argument('--seq_len', type=int, default=7)
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--test', action='store_true')
parser.add_argument('--log_dir', default='./log/')

parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--output_dim', type=int, default=1)

parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--use_gpu', action='store_true')

