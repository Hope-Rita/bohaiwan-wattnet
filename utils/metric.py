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
    sensors_metric = list()
    for i in range(y.shape[-1]):
        _,time,_ = y.shape
        time_list = list()
        for j in range(time):
            m = all_metric(y[:, j:j+1, i].reshape(-1), pred[:, j:j+1, i].reshape(-1))
            time_list.append(m)
        print(f'sensor-{sensor_name[i]}:', time_list)
        sensors_metric.append(time_list)
    return sensors_metric
