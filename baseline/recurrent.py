import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from dataset import get_data
from utils.calculate_metric import calculate_metrics
from utils.config import Config
from baseline.recurrent_fusion import RNNFusion, GRUFusion, LSTMFusion
from baseline.recurrent_seq import RNNSeqPredict, GRUSeqPredict, LSTMSeqPredict
from utils import normalization
from utils.load_config import get_Parameter
from utils.metric import RMSELoss
from utils import data_process
from utils.save import xw_toExcel
from wattnet_predict import get_dataloader, train_model

config_path = '../config.json'
conf = Config(config_path)
# 加载模型相关参数
num_workers, batch_size, epoch_num, learning_rate, save_model, load_model \
    = conf.get_config('recurrent-hyper-para',
                      inner_keys=['num-workers',
                                  'batch-size',
                                  'epoch-num',
                                  'learning-rate',
                                  'save-model',
                                  'load-model'
                                  ]
                      )
rnn_hidden_size = conf.get_config('recurrent-hyper-para', 'rnn-hidden-size')
gru_hidden_size = conf.get_config('recurrent-hyper-para', 'gru-hidden-size')
lstm_hidden_size = conf.get_config('recurrent-hyper-para', 'lstm-hidden-size')
# 选定运行的设备
if torch.cuda.is_available() and conf.get_config('device', 'use-gpu'):
    device = torch.device(conf.get_config('device', 'cuda'))
else:
    device = torch.device('cpu')
# 加载数据参数
pred_len, env_factor_num = conf.get_config('recurrent-hyper-para', inner_keys=['pred-len', 'env-factor-num'])
sensor_num, future_len = conf.get_config('data-parameters', inner_keys=['sensor-num', 'future-len'])
# 存放模型参数的路径
para_save_path = conf.get_config('model-weights-loc', 'local' if conf.get_config('run-on-local') else 'server')


def union_predict(model, x_train, y_train, x_test):
    """
    使用循环神经网络模型来进行联合预测
    @param model: 使用的具体模型
    @param x_train: 类型为 numpy 数组，形状为 m_train x (p + k)
    @param y_train: 类型为 numpy 数组，形状为 (m_train,)
    @param x_test: 类型为 numpy 数组，形状为 m_test x (p + k)
    @return: 预测结果，类型是 numpy 数组
    """
    # 加载数据
    data_loader, x_test = get_dataloader(x_train, y_train, x_test)

    if load_model:  # 加载已训练好的模型
        col = conf.get_config('predict-col')
        path = f'{para_save_path}/{col}_{model.name()}.pkl'
        model.load_state_dict(torch.load(path))
        print(f'从{path}处加载模型参数')
    else:  # 训练模型
        model = train_model(model, data_loader)

    # 将输出的结果进行处理并返回
    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    pred = pred.reshape(-1)
    return pred


def lstm_single_predict(x_train, y_train, x_test):
    model = LSTMSeqPredict(sensor_num=1, hidden_size=lstm_hidden_size, future_len=future_len).to(device)
    return seq_predict(model, x_train, y_train, x_test)


def gru_single_predict(x_train, y_train, x_test):
    model = GRUSeqPredict(sensor_num=1, hidden_size=gru_hidden_size, future_len=future_len).to(device)
    return seq_predict(model, x_train, y_train, x_test)


def rnn_single_predict(x_train, y_train, x_test):
    model = RNNSeqPredict(sensor_num=1, hidden_size=rnn_hidden_size, future_len=future_len).to(device)
    return seq_predict(model, x_train, y_train, x_test)


def seq_predict(model, x_train, y_train, x_test):
    # 加载数据
    data_loader, x_test = get_dataloader(x_train, y_train, x_test)
    # 训练模型
    model = train_model(model, data_loader)

    # 将输出的结果进行处理并返回
    pred = model(x_test)
    pred = pred.data.to('cpu').numpy()
    return pred


def rnn_seq_predict(x_train, y_train, x_test):
    model = RNNSeqPredict(sensor_num=sensor_num, hidden_size=rnn_hidden_size, future_len=future_len).to(device)
    return seq_predict(model, x_train, y_train, x_test)


def gru_seq_predict(x_train, y_train, x_test):
    model = GRUSeqPredict(sensor_num=sensor_num, hidden_size=gru_hidden_size, future_len=future_len).to(device)
    return seq_predict(model, x_train, y_train, x_test)


def lstm_seq_predict(x_train, y_train, x_test):
    model = LSTMSeqPredict(sensor_num=sensor_num, hidden_size=lstm_hidden_size, future_len=future_len).to(device)
    return seq_predict(model, x_train, y_train, x_test)


# def train_model(model, data_loader):
#     rmse = RMSELoss()
#     opt = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)
#     min_loss = np.inf
#     min_epoch = 0
#
#     with trange(epoch_num) as t:
#
#         for epoch in t:
#
#             now_lr = opt.state_dict()['param_groups'][0]['lr']
#             t.set_description(f'[e: {epoch}, lr:{now_lr}]')
#             model.train()
#
#             train_loss = 0.0
#             for i, data in enumerate(data_loader):
#                 x, y = data
#
#                 with torch.set_grad_enabled(True):
#                     pred_y = model(x)
#                     loss = rmse(pred_y.squeeze(), y)
#                     train_loss += loss.item() * len(x)
#
#                     opt.zero_grad()
#                     loss.backward()
#                     opt.step()
#
#             train_loss /= len(data_loader.dataset)
#             if train_loss < min_loss:
#                 min_loss = train_loss
#                 min_epoch = epoch
#             t.set_postfix(ml=min_loss, me=min_epoch)
#
#             scheduler.step(loss)  # 更新学习率
#
#     if save_model:  # 保存模型
#         col = conf.get_config('predict-col')
#         path = f'{para_save_path}/{col}_{model.name()}.pkl'
#         torch.save(model.state_dict(), path)
#         print(f'模型参数已存储到{path}')
#
#     return model


# def get_dataloader(x_train, y_train, x_test):
#     if x_train.ndim == 2:  # 改变形状格式, 使其变成三维的
#         x_train = x_train.reshape(-1, x_train.shape[1], 1)
#         x_test = x_test.reshape(-1, x_test.shape[1], 1)
#
#     # 转换成 tensor
#     x_train = torch.from_numpy(x_train).float().to(device)
#     y_train = torch.from_numpy(y_train).float().to(device)
#     x_test = torch.from_numpy(x_test).float().to(device)
#
#     # 构建 DataSet 和 DataLoader
#     dataset = TensorDataset(x_train, y_train)
#     data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
#     return data_loader, x_test


if __name__ == '__main__':
    pred_target_filename = conf.predict_target
    sensor_name = conf.get_config('data-parameters', 'valid-sensors')
    x_train, y_train, x_val, y_val, x_test, y_test, normal_y = get_data(pred_target_filename, 'beihang')
    train_loader, val_loader, x_test = get_dataloader(x_train[...,:4], y_train, x_val[...,:4], y_val, x_test[...,:4])
    model = RNNSeqPredict(sensor_num=sensor_num, hidden_size=rnn_hidden_size, future_len=future_len).to(device)
    # model = GRUSeqPredict(sensor_num=sensor_num, hidden_size=gru_hidden_size, future_len=future_len).to(device)
    # model = LSTMSeqPredict(sensor_num=sensor_num, hidden_size=lstm_hidden_size, future_len=future_len).to(device)
    model = train_model(model, train_loader, val_loader, draw_loss_pic=True)
    pred = model(x_test)

    pred = data_process.reverse_section_normalization(pred, normal_y)
    y_test = data_process.reverse_section_normalization(y_test, normal_y)

    if save_model:  # 保存模型
        torch.save(model.state_dict(), para_save_path)
        print(f'模型参数已存储到{para_save_path}')

    data = calculate_metrics(pred.cpu(), y_test.cpu(), 'test')
    print(data)

    xw_toExcel(data, 'result/' + get_Parameter('model_name') + '/' + get_Parameter('model_name') + '-result.xlsx')
    np.save('result/' + get_Parameter('model_name') + '/' + get_Parameter('model_name') + '-pred.npy', pred)
    np.save('result/' + get_Parameter('model_name') + '/' + get_Parameter('model_name') + '-truth.npy', y_test)

    print('Process end.')