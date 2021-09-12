import numpy as np
import os

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from utils.load_config import get_Parameter
from utils.util import convert_to_gpu


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        print(self.mean,self.std)

    def transform(self, data):
        data_scale = (data - self.mean) / self.std
        #data = np.concatenate((data_scale, data[:, :, get_Parameter('input_size'):]), axis=2)
        return data_scale

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MaxMinScaler:
    def __init__(self, Max, Min, need_transform):
        if not need_transform:
            self.Max = Max
            self.Min = Min
        else:
            self.Max = np.expand_dims(Max, 1).repeat(need_transform, axis=1)
            self.Min = np.expand_dims(Min, 1).repeat(need_transform, axis=1)

    def transform(self, data):
        print(data.shape)
        data_scale = (data - self.Min) / (self.Max - self.Min)
        #data = np.concatenate((data_scale, data[:, :, get_Parameter('input_size'):]), axis=2)
        return data_scale

    def inverse_transform(self, data):
        max_a = convert_to_gpu(torch.from_numpy(self.Max))
        min_a = convert_to_gpu(torch.from_numpy(self.Min))
        return (data * (max_a - min_a)) + min_a

class FeatureDataSet(Dataset):
    def __init__(self, inputs, target, covariate):
        self.inputs = inputs
        self.target = target
        self.covariate = covariate

    def __getitem__(self, item):
        return self.inputs[item], self.target[item], self.covariate[item]

    def __len__(self):
        return len(self.inputs)

def load_my_data(data_dirs, batch_size):
    loader = {}
    data = {}
    phases = ['train', 'val', 'test']
    data_all = None

    for phase in phases:
        cat_data = np.load(os.path.join(data_dirs, phase + '.npy'))
        s = cat_data.shape[0]
        x_data = cat_data[:, :get_Parameter('window'), :get_Parameter('input_size')].reshape(s, get_Parameter('window'), 1, -1)
        y_data = cat_data[:, get_Parameter('window'):, :get_Parameter('input_size')].reshape(s, get_Parameter('target'), 1, -1)
        data['x_' + phase] = np.swapaxes(x_data, 2, 3)
        data['y_' + phase] = np.swapaxes(y_data, 2, 3)
        data['covariate_' + phase] = cat_data[:, :, -get_Parameter('covariate_size'):]
        if data_all is not None:
            data_all = np.vstack((data_all, cat_data[:, :, :get_Parameter('input_size')]))
        else:
            data_all = cat_data[:, :, :get_Parameter('input_size')]
    s, T, N = data_all.shape
    print(data_all.shape)
    if get_Parameter('normalized') == 1:
        scaler = StandardScaler(mean=np.mean(data_all.reshape(-1,N),axis=0), std=np.std(data_all.reshape(-1,N),axis=0))
    elif get_Parameter('normalized') == 2:
        scaler = MaxMinScaler(Max=np.max(data_all.reshape(-1,N),axis=0), Min=np.min(data_all.reshape(-1,N),axis=0), need_transform=False)
    elif get_Parameter('normalized') == 3:
        scaler = MaxMinScaler(Max=np.amax(data_all, (0, 1, 3)), Min=np.amin(data_all, (0, 1, 3)), need_transform=1)
    else:
        pass
    loader['scaler'] = scaler
    for phase in phases:
        data['x_' + phase] = scaler.transform(data['x_' + phase])
        data['y_' + phase] = scaler.transform(data['y_' + phase])
        loader[phase] = DataLoader(FeatureDataSet(inputs=data['x_' + phase], target=data['y_' + phase], covariate=data['covariate_' + phase]), batch_size,
                                   shuffle=True, drop_last=True)
    return loader, scaler

