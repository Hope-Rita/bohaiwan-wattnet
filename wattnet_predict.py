import numpy as np
import torch
import time
import os
import copy
from torch import optim
from torch.utils.data import TensorDataset
from visdom import Visdom
from utils.config import Config
from models.wattnet import WATTNet
from utils.metric import RMSELoss
from utils.background_loader import BackgroundLoader
from utils.draw_pic import train_process_pic


conf = Config()
# 加载模型训练的相关配置
num_workers, batch_size, epoch_num, learning_rate, save_model, load_model, visdom_env, save_name, show_attn \
    = conf.get_config('model-config',
                      inner_keys=['num-workers',
                                  'batch-size',
                                  'epoch-num',
                                  'learning-rate',
                                  'save-model',
                                  'load-model',
                                  'visdom-env',
                                  'save-name',
                                  'show-attn'
                                  ]
                      )
# 加载模型超参
depth, n_repeat = conf.get_config('model-hyper-para', inner_keys=['depth', 'n-repeat'])

# 选定运行的设备
if torch.cuda.is_available() and conf.get_config('device', 'use-gpu'):
    device = torch.device(conf.get_config('device', 'cuda'))
else:
    device = torch.device('cpu')
# 加载模型和数据的参数
pred_len, future_len, sensor_num = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-len', 'sensor-num'])
# 存放模型参数的路径
para_save_path = conf.get_config('model-weights-loc', conf.run_location)


def wattnet_predict(x_train, y_train, x_val, y_val, x_test):
    model = WATTNet(series_len=pred_len,
                    in_dim=sensor_num,
                    out_dim=future_len,
                    depth=depth,
                    n_repeat=n_repeat,
                    show_attn_alpha=show_attn
                    ).to(device)
    print(f'载入模型:{model.name}, depth: {depth}, n_repeat: {n_repeat}')
    train_loader, val_loader, x_test = get_dataloader(x_train, y_train, x_val, y_val, x_test)

    path = os.path.join(para_save_path, f'{model.name}-{save_name}.pkl')
    if load_model:  # 加载模型
        model.load_state_dict(torch.load(path, map_location=conf.get_config('device', 'cuda')))
        print(f'从{path}处加载模型参数')
    else:  # 训练模型
        model = train_model(model, train_loader, val_loader, draw_loss_pic=True)

    # 进行预测
    pred = model(x_test)

    if save_model:  # 保存模型
        torch.save(model.state_dict(), path)
        print(f'模型参数已存储到{path}')

    return pred.data.to('cpu').numpy()


def train_model(model, train_loader, val_loader, draw_loss_pic=False):
    rmse = RMSELoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, threshold=1e-3, min_lr=1e-6)
    min_loss, min_epoch = np.inf, 0
    min_val_loss = np.inf
    # best_model = None
    train_loss_record, val_loss_record = [], []
    viz = Visdom(env=visdom_env)  # 使用 visdom 进行实时可视化

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
        min_loss, min_epoch = (train_loss, epoch) if train_loss < min_loss else (min_loss, min_epoch)

        # 验证阶段
        val_loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                val_loss += rmse(model(x), y).item() * len(x)
        val_loss /= len(val_loader.dataset)
        val_loss_record.append(val_loss)
        if epoch >= 10 and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
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
        train_process_pic(train_loss_record, val_loss_record, title=f'Train Process {save_name}')
    return model


def get_dataloader(x_train, y_train, x_val, y_val, x_test):
    # 转换成 tensor
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_val = torch.from_numpy(x_val).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)

    # 构建 DataSet 和 DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = BackgroundLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = BackgroundLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, x_test
