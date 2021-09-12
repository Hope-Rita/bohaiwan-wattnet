import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden, num_layers, output_len=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_size = 1
        self.dropout = dropout
        self.output_len = output_len
        self.embedding = nn.Sequential(nn.Linear(self.input_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=self.hidden, num_layers=self.num_layers, batch_first=True,
                          dropout=dropout)
        self.linear = nn.Linear(self.hidden * self.num_layers, self.input_size)

    def forward(self, input):
        """

        :param input:[batch, T, N]
        :return:
        """

        output = []

        for i in range(self.output_len):
            batch, T, N = input.size()
            input1 = input.permute(0, 2, 1)
            input1 = input1.reshape(batch * N, T, -1)
            input1 = self.embedding(input1)
            out, h = self.lstm(input1)
            h = h[0].permute(1, 0, 2).contiguous().view(batch * N, -1)
            l = self.linear(h)
            out = l.unsqueeze(2)
            out = out.reshape(batch, N, -1).permute(0, 2, 1)
            input = torch.cat([input[:, 1:, :], out], dim=1)
            output.append(out)

        return torch.cat(output, dim=1)


class LSTM_single(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM_single, self).__init__()
        self.input_size = input_size
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        :param x: [batch, T, N]
        :return: [batch, N, 1]
        """
        x = x[:, :, :self.input_size]
        batch, T, N = x.size()
        x = x.transpose(1, 2).contiguous().view(-1, T).unsqueeze(2)
        out, h = self.rnn(x)
        l = self.linear(h[0]).squeeze(0).view(batch, N, 1)
        return l.transpose(1, 2)


class LSTM_Trans(nn.Module):
    def __init__(self, input_size, hidden, num_layers=1):
        super(LSTM_Trans, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_size = input_size
        self.embedding = nn.Sequential(nn.Linear(input_size, 200), nn.ReLU(), nn.Linear(200, 200))
        # self.embedding = nn.Linear(input_size, 200)
        # nn.init.xavier_uniform(self.embedding.weight)
        # self.U = nn.Parameter(torch.FloatTensor(input_size, 100))
        # self.V = nn.Parameter(torch.FloatTensor(100, 200))
        # nn.init.xavier_uniform(self.U)
        # nn.init.xavier_uniform(self.V)
        self.rnn = nn.LSTM(input_size=200, hidden_size=self.hidden, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden, input_size)

    def forward(self, input):
        """

        :param input:[batch, T, N]
        :return:
        """
        x = input[:, :, :self.input_size]
        # self.embedding = torch.matmul(self.U, self.V)
        # x = torch.matmul(x, self.embedding)
        x = self.embedding(x)
        out, h = self.rnn(x)
        l = self.linear(h[0])
        return l.permute(1, 2, 0)
