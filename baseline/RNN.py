import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size, hidden, num_layers, output_len=1, dropout = 0.2):
        super(RNN, self).__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.input_size = 1
        self.output_len = output_len
        self.dropout = dropout
        self.embedding = nn.Sequential(nn.Linear(self.input_size, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.rnn = nn.RNN(input_size=hidden, hidden_size=self.hidden, num_layers=self.num_layers, batch_first=True,dropout=dropout)
        self.linear = nn.Linear(self.hidden * self.num_layers, self.input_size)

    def forward(self, input):
        """

        :param input:[batch, T, N]
        :return:
        """
        output = []

        for i in range(self.output_len):
            batch, T, N = input.size()
            input1 = input.permute(0,2,1)
            input1 = input1.reshape(batch*N, T, -1)
            input1 = self.embedding(input1)
            out, h = self.rnn(input1)
            h = h.permute(1, 0, 2).contiguous().view(batch*N, -1)
            l = self.linear(h)
            out = l.unsqueeze(2)
            out = out.reshape(batch,N,-1).permute(0,2,1)
            input = torch.cat([input[:, 1:, :],out],dim=1)
            output.append(out)

        return torch.cat(output,dim=1)