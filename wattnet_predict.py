import numpy as np
import torch
import time
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils.config import Config
from models.wattnet import WATTNet
from utils.metric import RMSELoss


conf = Config()
# 加载模型相关参数
num_workers, batch_size, epoch_num, learning_rate, save_model, load_model \
    = conf.get_config('model-parameters',
                      inner_keys=['num-workers',
                                  'batch-size',
                                  'epoch-num',
                                  'learning-rate',
                                  'save-model',
                                  'load-model'
                                  ]
                      )

# 选定运行的设备
if torch.cuda.is_available() and conf.get_config('device', 'use-gpu'):
    device = torch.device(conf.get_config('device', 'cuda'))
else:
    device = torch.device('cpu')
# 加载模型和数据的参数
pred_len, future_len, sensor_num = conf.get_config('data-parameters', inner_keys=['pred-len', 'future-len', 'sensor-num'])
# 存放模型参数的路径
para_save_path = conf.get_config('model-paras', 'local' if conf.get_config('run-on-local') else 'server')


def wattnet_predict(x_train, y_train, x_test):
    model = WATTNet(in_dim=sensor_num, out_dim=future_len).to(device)
    data_loader, x_test = get_dataloader(x_train, y_train, x_test)
    model = train_model(model, data_loader)
    pred = model(x_test)
    return pred.data.to('cpu').numpy()


def train_model(model, data_loader):
    rmse = RMSELoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)
    min_loss = np.inf
    min_epoch = 0

    for epoch in range(epoch_num):

        now_time = time.strftime('%H:%M:%S')
        print(f'[{now_time} epoch: {epoch}, lr:{learning_rate}]', end=' ')
        model.train()

        train_loss = 0.0
        for i, data in enumerate(data_loader):
            x, y = data

            with torch.set_grad_enabled(True):
                pred_y = model(x)
                loss = rmse(pred_y, y)
                train_loss += loss.item() * len(x)

                opt.zero_grad()
                loss.backward()
                opt.step()

        train_loss /= len(data_loader.dataset)
        if train_loss < min_loss:
            min_loss = train_loss
            min_epoch = epoch
        print(f'min_loss: {min_loss}, min_epoch: {min_epoch}')

        scheduler.step(loss)  # 更新学习率

    if save_model:  # 保存模型
        col = conf.get_config('predict-col')
        path = f'{para_save_path}/{col}_{model.name()}.pkl'
        torch.save(model.state_dict(), path)
        print(f'模型参数已存储到{path}')

    return model


def get_dataloader(x_train, y_train, x_test):
    # 转换成 tensor
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    # 构建 DataSet 和 DataLoader
    dataset = TensorDataset(x_train, y_train)
    data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return data_loader, x_test
