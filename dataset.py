import numpy as np
import pandas as pd
from utils import data_process
from utils.config import Config


conf = Config()

# 加载运行配置
pred_len, future_len, move_interval, valid_sensors = \
    conf.get_config('data-parameters', inner_keys=['pred-len', 'future-len', 'move-interval', 'valid-sensors'])
print(f'载入dataset模块, pred: {pred_len}, future: {future_len}, interval: {move_interval}')


def get_data(filename):

    def is_valid_index(idx):
        valid = True
        for j in range(idx, idx + pred_len + future_len):
            valid = valid and (hours[j // 6])
        return valid

    # 可用的 hour
    hours = []
    label_df = pd.read_csv(conf.get_data_loc('label'), header=None)
    for i in range(744):
        flag = True
        for sensor in valid_sensors:
            flag = flag and (label_df.loc[sensor, i] == 1)
        hours.append(flag)

    print(f'从 {filename} 中加载数据')
    raw_data = np.load(filename)  # (m, sensor_num)
    x, y = [], []
    for i in range(0, len(raw_data) - pred_len - future_len, move_interval):
        if is_valid_index(i):
            x.append(raw_data[i: i + pred_len, valid_sensors])
            y.append(raw_data[i + pred_len: i + pred_len + future_len, valid_sensors])

    x = data_process.section_normalization(np.array(x))
    y, normal_y = data_process.section_normalization_with_normalizer(np.array(y))
    train_size = int(0.8 * len(x))
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:], normal_y
