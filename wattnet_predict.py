import numpy as np
import torch
import time
from torch import optim
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from visdom import Visdom
from utils.config import Config
from models.wattnet import WATTNet
from utils.metric import RMSELoss
from utils.background_loader import BackGroundLoader
from utils.draw_pic import train_process_pic


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
pred_len, future_len, sensor_num = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-len', 'sensor-num'])
# 存放模型参数的路径
para_save_path = conf.get_config('model-paras', 'local' if conf.get_config('run-on-local') else 'server')


def wattnet_predict(x_train, y_train, x_test):
    model = WATTNet(in_dim=sensor_num, out_dim=future_len).to(device)
    train_loader, val_loader, x_test = get_dataloader(x_train, y_train, x_test)
    model = train_model(model, train_loader, val_loader, draw_loss_pic=True)
    pred = model(x_test)
    return pred.data.to('cpu').numpy()


def train_model(model, train_loader, val_loader, draw_loss_pic=False):
    rmse = RMSELoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)
    min_loss = np.inf
    min_epoch = 0
    train_loss_record, val_loss_record = [], []

    # 使用 visdom 进行实时可视化
    viz = Visdom(env='Train process-2')

    for epoch in range(epoch_num):

        now_time = time.strftime('%H:%M:%S')
        now_lr = opt.state_dict()['param_groups'][0]['lr']
        print(f'[{now_time} epoch: {epoch}, lr:{now_lr}]', end=' ')
        model.train()

        # 训练阶段
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            x, y = data

            with torch.set_grad_enabled(True):
                pred_y = model(x)
                loss = rmse(pred_y, y)
                train_loss += loss.item() * len(x)

                opt.zero_grad()
                loss.backward()
                opt.step()

        train_loss /= len(train_loader.dataset)
        train_loss_record.append(train_loss)
        if train_loss < min_loss:
            min_loss = train_loss
            min_epoch = epoch

        # 验证阶段
        val_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                val_loss += rmse(model(x), y).item() * len(x)
        val_loss /= len(val_loader.dataset)
        val_loss_record.append(val_loss)
        scheduler.step(val_loss)  # 更新学习率
        print(f'train_loss: {train_loss}, valid_loss: {val_loss}, min_loss: {min_loss}, min_epoch: {min_epoch}')
        viz.line(Y=np.array([train_loss, val_loss]).reshape(1, 2),
                 X=np.array([epoch, epoch]).reshape(1, 2),
                 win='line',
                 update=(None if epoch == 0 else 'append'),
                 opts={'legend': ['train_loss', 'valid_loss']}
                 )

    # 绘制 loss 变化图
    if draw_loss_pic:
        train_process_pic(train_loss_record, val_loss_record, title='Train Process')

    if save_model:  # 保存模型
        col = conf.get_config('predict-col')
        path = f'{para_save_path}/{col}_{model.name()}.pkl'
        torch.save(model.state_dict(), path)
        print(f'模型参数已存储到{path}')

    return model


def get_dataloader(x_train, y_train, x_test):
    # 划分训练集和验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125)
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_val: {x_val.shape}, y_val: {y_val.shape}')
    # 转换成 tensor
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_val = torch.from_numpy(x_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    # 构建 DataSet 和 DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = BackGroundLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = BackGroundLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, x_test
