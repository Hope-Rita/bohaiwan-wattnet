import os
import pickle
import zipfile

import torch
import numpy as np
import pandas as pd

import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.sparse.linalg import eigs

from utils.load_config import get_Parameter


# convert data from cpu to gpu, accelerate the running speed
def convert_to_gpu(data):
    if torch.cuda.is_available():
        data = data.cuda(get_Parameter('cuda'))
    return data

# saves parameters of the model
def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)


def load_graph_data(graph_filename):
    # adj_mx = np.load(graph_filename)
    adj_mx = np.eye(get_Parameter('input_size'))
    return adj_mx


def create_single_kernel(adj, type):
    if type == 'random':
        return calculate_random_walk_matrix(adj).T
    elif type == 'cheb':
        return calculate_scaled_laplacian(adj, lambda_max=None)
    else:
        return calculate_scaled_laplacian(adj, lambda_max=None)


# DCRNN
def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


# lamda_max = 2 is first approx
def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    # L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    L = sp.coo_matrix(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    print(lambda_max)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()


# STGCN
def cheb_poly_approx(L, Ks, n):
    """
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param Ks: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    """
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))

    if Ks > 1:
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(Ks - 2):
            Ln = np.mat(2 * L * L1 - L0)
            L_list.append(np.copy(Ln))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
        # L_lsit [Ks, n*n], Lk [n, Ks*n]
        return np.concatenate(L_list, axis=-1)
    elif Ks == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{Ks}".')


def first_approx(W, n):
    """
    1st-order approximation function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, n_route].
    """
    # A = W + np.identity(n)
    A = W
    d = np.sum(A, axis=1)
    sinvD = np.sqrt(np.mat(np.diag(d)).I)
    # refer to Eq.5
    return np.mat(np.identity(n) + sinvD * A * sinvD)


def build_sparse_matrix(L):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    # shape = L.shape
    # i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
    # v = torch.FloatTensor(L.data)
    # return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    L = L.tocoo()
    indices = np.column_stack((L.row, L.col))
    # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
    indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
    L = torch.sparse_coo_tensor(indices.T, L.data, L.shape)
    return L


def create_kernel(L, Ks):
    L = build_sparse_matrix(L)

    def concat(x, x_):
        return torch.cat([x, x_], dim=0)
    # all = []
    # all.append(L)
    # x0 = torch.eye(L.shape[0])
    # all = []
    # x = x0
    # all.append(x)
    # x1 = torch.sparse.mm(L, x0)
    # x = concat(x, x1)
    # all.append(x1)
    # for k in range(2, Ks + 1):
    #     x2 = 2 * torch.sparse.mm(L, x1) - x0
    #     x = concat(x, x2)
    #     all.append(x2)
    #     x1, x0 = x2, x1
    return L


def record_predict_result(pre, tar):
    pre, tar = pre[:, 0, :], tar[:, 0, :]
    import pandas as pd
    if get_Parameter('input_size') == 282:
        station_columns = pd.read_csv(
            '/mnt/windows-E/qyn/traffic-data-preprocessor/IC-record-preprocessor/data/final-data2/bus-pickup-data-filter.csv',
            header=0, index_col=0).columns
    else:
        station_columns = pd.read_csv(
            '/mnt/windows-E/qyn/traffic-data-preprocessor/IC-record-preprocessor/data/final-data2/subway-pickup-data-filter.csv',
            header=0, index_col=0).columns
    # coprediction_columns = bus_station_columns + subway_station_columns
    index_col = pd.date_range('2017-09-07 00:00:00', '2017-09-30 23:00:00', freq='1H')
    prediction = pd.DataFrame(pre, columns=station_columns, index=index_col)
    target = pd.DataFrame(tar, columns=station_columns, index=index_col)
    prediction.to_csv(
        'data/' + get_Parameter('model_name') + '_' + str(get_Parameter('input_size')) + '_' + 'predict_result.csv',
        header=True, index=True)
    target.to_csv('data/' + get_Parameter('model_name') + '_' + str(get_Parameter('input_size')) + '_' + 'target.csv',
                  header=True, index=True)
