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
    rmse_val = rmse(y, pred)
    mae_val = mae(y, pred)
    mape_val = mape(y, pred)
    pcc_val = pcc(y, pred)

    return {
        'RMSE': f'{round(rmse_val, 4)}',
        'MAE': f'{round(mae_val, 4)}',
        'MAPE': round(mape_val, 4),
        'PCC': round(pcc_val, 4)
    }


def metric_for_each_sensor(y, pred, sensor_name):
    print('\n预测指标')
    sensors_metric = []
    for i in range(y.shape[-1]):
        m = all_metric(y[:, :, i].reshape(-1), pred[:, :, i].reshape(-1))
        sensors_metric.append(m)
        print(f'sensor-{sensor_name[i]}:', m)
    return sensors_metric
