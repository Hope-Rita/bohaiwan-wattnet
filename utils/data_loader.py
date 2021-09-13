from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from utils.data_container import FeatureDataSet
from utils.load_config import get_Parameter
import torch
import pandas as pd
import random

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
        data_raw = data[:, :, :get_Parameter('input_size')]
        data_scale = (data_raw - self.mean)/self.std
        data = np.concatenate((data_scale, data[:, :, get_Parameter('input_size'):]), axis=2)

        return data

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MaxMinScaler:
    def __init__(self, Max, Min):
        self.Max = Max
        self.Min = Min

    def transform(self, data):
        data_raw = data[:, :, :get_Parameter('input_size')]
        data_scale = (data_raw - self.Min) / (self.Max - self.Min)
        data = np.concatenate((data_scale, data[:, :, get_Parameter('input_size'):]), axis=2)
        return data

    def inverse_transform(self, data):
        return (data * (self.Max - self.Min)) + self.Min


def load_data(data_dirs, batch_size, normalized=1):

    loader = {}
    data = {}
    phases = ['train', 'val', 'test']
    data_all = None
    str = '-36-15'
    for phase in phases:
        cat_data = np.load(os.path.join(data_dirs, phase + str + '.npy'))
        data['x_' + phase] = cat_data[:, :get_Parameter('window'), :get_Parameter('input_size')]
        data['y_' + phase] = cat_data[:, get_Parameter('window'):, :get_Parameter('input_size')]
        data['covariate_' + phase] = cat_data[:, :get_Parameter('window'), get_Parameter('input_size'):]
        # print(data['x_train'].shape)
        if data_all is not None:
            data_all = np.vstack((data_all, cat_data[:,:,:get_Parameter('input_size')]))
        else:
            data_all = cat_data[:, :, :get_Parameter('input_size')]
    str = ''
    mean = np.load(os.path.join(data_dirs, 'mean' + str + '.npy'))
    std = np.load(os.path.join(data_dirs, 'std' + str + '.npy'))
    max = np.load(os.path.join(data_dirs, 'max' + str + '.npy'))
    min = np.load(os.path.join(data_dirs, 'min' + str + '.npy'))
    # print("mean:",mean)
    # print("std:",std)
    if get_Parameter('normalized'):
        if normalized == 1:
            scaler = StandardScaler(mean=mean.astype('float32'), std=std.astype('float32'))
        elif normalized == 2:
            scaler = MaxMinScaler(Max=max.astype('float32'), Min=min.astype('float32'))
        for phase in phases:
            data['x_' + phase] = scaler.transform(data['x_' + phase])
            data['y_' + phase] = scaler.transform(data['y_' + phase])
    else:
        scaler = None

    for phase in phases:
        if phase == 'train':
            loader[phase] = DataLoader(FeatureDataSet(inputs=data['x_' + phase], target=data['y_' + phase], covariate=data['covariate_' + phase]), batch_size,
                                   shuffle=True, drop_last=False)
        else:
            loader[phase] = DataLoader(FeatureDataSet(inputs=data['x_' + phase], target=data['y_' + phase], covariate=data['covariate_' + phase]), batch_size,
                                   shuffle=False, drop_last=False)
    return loader, scaler


