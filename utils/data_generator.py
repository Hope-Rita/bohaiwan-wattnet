import torch
import numpy as np
import pandas as pd
import os


class Data_utility():
    def __init__(self, data_path, train, valid, window, target, horizon=0, gap = 0):
        self.raw_data = pd.read_csv(data_path, header=0, index_col=0,encoding='utf-8').values
        # self.raw_data = np.loadtxt(data_path, delimiter=',')
        self.data = self.raw_data
        self.length, self.dimension = self.raw_data.shape
        self.window = window
        self.horizon = horizon
        self.target = target
        self.gap = gap
        # normalized
        # temp = self.normalized()
        # self.data = self.raw_data
        # self.data[:,4:] = temp[:,4:]
        # self.data = self.normalized()
        self.normalized()
        self._split(int(train * self.length), int((train + valid) * self.length), self.length)

    # def indice_from_data(self, window, target):
    #     seq_data = [(i, i + window + target) for i in range(self.length - window - target + 1)]
    #     np.random.shuffle(seq_data)
    #     return seq_data

    def normalized(self):
        self.mean = np.mean(self.data,axis=0)
        self.std = np.std(self.data,axis=0)
        self.max = np.max(self.data,axis=0)
        self.min = np.min(self.data,axis=0)
        print(self.mean.shape)
        # return (self.raw_data - self.mean)/self.std
        return (self.data - self.min) / (self.max - self.min)
    # 数据集的划分
    def _split(self, train, valid, all):
        train_set = range(self.window + self.target + self.gap - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, all)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    # 数据集划分的迭代流程
    def _batchify(self, index_set):
        n = len(index_set)
        X = torch.zeros((n, self.window, self.dimension))
        Y = torch.zeros((n, self.target, self.dimension))
        Z = torch.zeros((n, self.target, self.dimension))
        for i in range(n):
            end_index = index_set[i] - self.target + 1 - self.gap
            start_index = end_index - self.window
            X[i, :, :] = torch.from_numpy(self.data[start_index:end_index, :])
            Y[i, :, :] = torch.from_numpy(self.data[end_index+self.gap: index_set[i] + 1, :])
            # Z[i, :, :] = torch.from_numpy(self.data[end_index+self.gap - 24: index_set[i] + 1 -24, :])
        # return [X, Z, Y]
        return [X, Y]

def get_train_data(from_path, to_path):
    d = Data_utility(data_path=from_path, train=0.7, valid=0.1, window=27, target=15, gap=0)
    train = np.concatenate(d.train, axis=1)
    valid = np.concatenate(d.valid, axis=1)
    test = np.concatenate(d.test, axis=1)
    np.save(os.path.join(to_path, 'train-27-15.npy'), train[:,:,:7])
    np.save(os.path.join(to_path, 'val-27-15.npy'), valid[:,:,:7])
    np.save(os.path.join(to_path, 'test-27-15.npy'), test[:,:,:7])
    np.save(os.path.join(to_path, 'mean.npy'), d.mean[:4])
    np.save(os.path.join(to_path, 'std.npy'), d.std[:4])
    np.save(os.path.join(to_path, 'max.npy'), d.max[:4])
    np.save(os.path.join(to_path, 'min.npy'), d.min[:4])
    print('construct {} finished'.format(from_path))
    print('train data shape {}'.format(train.shape))
    print('valid data shape {}'.format(valid.shape))
    print('test data shape {}'.format(test.shape))


import pandas as pd

def xlsx_to_csv_pd():
    data_xls = pd.read_excel('../data/data.xlsx', index_col=0)
    data_xls.to_csv('../data/data.csv', encoding='utf-8')


if __name__ == '__main__':
    # xlsx_to_csv_pd()
    get_train_data(from_path='../data/data.csv', to_path='../data/27-15/')
    # print("data_generator")