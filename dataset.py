import numpy as np
from utils.config import Config


conf = Config()

# 加载运行配置
pred_len, future_len, move_interval = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-len', 'move-interval'])
print(f'\n配置文件：{conf.path}，载入dataset模块, pred: {pred_len}, future: {future_len}, interval: {move_interval}')


def get_data(filename):
    raw_data = np.load(filename)  # (m, sensor_num)
    x, y = [], []
    for i in range(0, len(raw_data) - pred_len - future_len, move_interval):
        x.append(raw_data[i: i + pred_len])
        y.append(raw_data[i + pred_len: i + pred_len + future_len])

    x, y = np.array(x), np.array(y)
    train_size = int(0.8 * len(x))
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]
