import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn import metrics
from torch import nn


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return torch.sqrt(self.mse(y_hat, y))


def mae(y, pred):
    return metrics.mean_absolute_error(y, pred)


def mse(y, pred):
    return metrics.mean_squared_error(y, pred)


def rmse(y, pred):
    return np.sqrt(mse(y, pred))


def mape(y, pred):
    return 100 * np.sum(np.abs((y - pred) / y)) / len(y)


def pcc(y, pred):
    return pearsonr(y, pred)[0]


def all_metric(y, pred):
    return {
        'RMSE': rmse(y, pred),
        'MAE': mae(y, pred),
        'MAPE': mape(y, pred),
        'PCC': pcc(y, pred)
    }


def metric_for_each_sensor(y, pred, sensor_name):
    print('\n预测指标')
    for i in range(y.shape[2]):
        print(f'sensor-{sensor_name[i]}:', all_metric(y[:, :, i].reshape(-1), pred[:, :, i].reshape(-1)))
