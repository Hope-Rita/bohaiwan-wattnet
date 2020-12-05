import numpy as np
from utils import data_process
from utils.config import Config


conf = Config()

# 加载运行配置
pred_len, future_len, move_interval = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-len', 'move-interval'])
print(f'载入dataset模块, pred: {pred_len}, future: {future_len}, interval: {move_interval}')


def get_data(filename):
    print(f'从 {filename} 中加载数据')
    raw_data = np.load(filename)  # (m, sensor_num)
    x, y = [], []
    for i in range(0, len(raw_data) - pred_len - future_len, move_interval):
        x.append(raw_data[i: i + pred_len])
        y.append(raw_data[i + pred_len: i + pred_len + future_len])

    x = data_process.section_normalization(np.array(x))
    y, normal_y = data_process.section_normalization_with_normalizer(np.array(y))
    train_size = int(0.8 * len(x))
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:], normal_y
