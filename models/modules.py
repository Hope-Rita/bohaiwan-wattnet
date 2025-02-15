import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from utils.draw_pic import show_attention_matrix


class AttentionBlock(nn.Module):
    def __init__(self, in_channels: int, key_size: int, value_size: int):
        """Attention block without masking"""
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(key_size, key_size)
        self.linear_keys = nn.Linear(key_size, key_size)
        self.linear_values = nn.Linear(value_size, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, x_in, show_alpha=None):
        """
        :param x_in: (N, W, C)
        :param show_alpha: 若为 None, 则表示不输出 alpha, 否则表示输出的层数
        """
        bs = x_in.size(0)
        w_dim = x_in.size(1)
        x_orig = x_in

        x_in = x_in.reshape((bs, -1))
        keys = self.linear_keys(x_in)
        keys = keys.reshape((bs, w_dim, -1))  # `N, W, key_size`

        query = self.linear_query(x_in)
        query = query.reshape((bs, w_dim, -1))  # `N, W, key_size`

        values = self.linear_values(x_in)
        values = values.reshape((bs, w_dim, -1))  # `N, W, value_size`

        alphas = torch.bmm(query, torch.transpose(keys, 1, 2))  # `N, W, W`
        alphas = functional.softmax(alphas / self.sqrt_key_size, dim=1)  # `N, W, W`
        res = torch.bmm(alphas, values)  # `N, W, value_size`
        res = torch.sigmoid(res)

        if show_alpha:
            alphas_to_show = alphas.cpu().detach().numpy().mean(axis=0)
            show_attention_matrix(alphas_to_show, show_alpha)

        return res + x_orig


class GatedBlock(nn.Module):
    def __init__(self, dilation: int, w_dim: int, seq_in, seq_out):
        """Gated block with sigmoid/tanh gates."""
        super().__init__()
        self.dilation = dilation
        self.res = nn.Linear(seq_in,seq_out)
        self.tanh_conv = nn.Conv2d(w_dim, w_dim, kernel_size=(2, 1), dilation=(dilation, 1), groups=w_dim)
        self.sigmoid_conv = nn.Conv2d(w_dim, w_dim, kernel_size=(2, 1), dilation=(dilation, 1), groups=w_dim)
        self.out_conv = nn.Conv2d(w_dim, w_dim, kernel_size=1, groups=w_dim)

    def forward(self, x_in):
        B,N,T,F = x_in.shape
        x_tanh, x_sigmoid = self.tanh_conv(x_in), self.sigmoid_conv(x_in)
        x_gate = torch.tanh(x_tanh) * torch.sigmoid(x_sigmoid)
        x_res = self.res(x_in.transpose(2,3).reshape(-1,T))
        x_res = x_res.reshape(B,N,F,-1).transpose(2,3)
        x_out = self.out_conv(x_gate + x_res)
        return x_out  # (N, w_dim, H-dilation, ebd_dim)
