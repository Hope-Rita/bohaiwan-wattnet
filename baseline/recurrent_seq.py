import torch
from torch import nn
from utils.config import Config


conf = Config()
device = torch.device(conf.get_config('device', 'cuda') if torch.cuda.is_available() else 'cpu')


class RNNSeqPredict(nn.Module):

    def __init__(self, sensor_num, hidden_size, future_len):
        super(RNNSeqPredict, self).__init__()
        self.future_len = future_len
        self.hidden_size = hidden_size
        self.encoder = nn.RNN(input_size=sensor_num, hidden_size=hidden_size)
        self.decoder = nn.RNN(input_size=hidden_size, hidden_size=hidden_size)
        self.out_fc = nn.Linear(in_features=hidden_size, out_features=sensor_num)

    def forward(self, x):
        batch = x.shape[0]
        _, hn = self.encoder(x.transpose(0, 1))
        dec_input = torch.cat([hn, torch.zeros(self.future_len - 1, batch, self.hidden_size, device=device)])
        dec_output, _ = self.decoder(dec_input)
        dec_output = dec_output.transpose(0, 1)
        return self.out_fc(dec_output)


class GRUSeqPredict(nn.Module):

    def __init__(self, sensor_num, hidden_size, future_len):
        super(GRUSeqPredict, self).__init__()
        self.future_len = future_len
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size=sensor_num, hidden_size=hidden_size)
        self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.out_fc = nn.Linear(in_features=hidden_size, out_features=sensor_num)

    def forward(self, x):
        batch = x.shape[0]
        _, hn = self.encoder(x.transpose(0, 1))
        dec_input = torch.cat([hn, torch.zeros(self.future_len - 1, batch, self.hidden_size, device=device)])
        dec_output, _ = self.decoder(dec_input)
        dec_output = dec_output.transpose(0, 1)
        return self.out_fc(dec_output)


class LSTMSeqPredict(nn.Module):

    def __init__(self, sensor_num, hidden_size, future_len):
        super(LSTMSeqPredict, self).__init__()
        self.future_len = future_len
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=sensor_num, hidden_size=hidden_size)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        self.out_fc = nn.Linear(in_features=hidden_size, out_features=sensor_num)

    def forward(self, x):
        batch = x.shape[0]
        _, (hn, _) = self.encoder(x.transpose(0, 1))
        dec_input = torch.cat([hn, torch.zeros(self.future_len - 1, batch, self.hidden_size, device=device)])
        dec_output, _ = self.decoder(dec_input)
        dec_output = dec_output.transpose(0, 1)
        return self.out_fc(dec_output)
