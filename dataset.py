import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import data_process
from utils.config import Config


config_path = 'config.json'
conf = Config(config_path)

# 加载运行配置
pred_len, future_gap, future_len, move_interval, valid_sensors = \
    conf.get_config('data-parameters', inner_keys=['pred-len',
                                                   'future-gap',
                                                   'future-len',
                                                   'move-interval',
                                                   'valid-sensors'
                                                   ]
                    )
print(f'载入dataset模块, pred: {pred_len}, gap: {future_gap}, future: {future_len}, interval: {move_interval}')


# 用于加载李慧的竞赛数据
def get_lihui_data(filename):

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

    return x, y

def get_beihang_data(filename):
    raw_data = pd.read_csv(filename, header=0, index_col=0, encoding='utf-8').values
    x, y, z = [], [], []
    for i in range(0, len(raw_data) - pred_len - future_len, move_interval):
        x.append(raw_data[i: i + pred_len, :4])
        y.append(raw_data[i + pred_len: i + pred_len + future_len, :4])
        z.append(raw_data[i + pred_len: i + pred_len + future_len, 4:])

    return x, y, z


# 用于加载南京的隧道数据
def get_nanjing_data(filename):
    frame = pd.read_csv(filename, parse_dates=True, index_col='time')
    x, y, pred_times = [], [], []

    for time in pd.date_range(frame.index[0], frame.index[-1], freq='H'):

        pred_time_start = time + pd.Timedelta(hours=(pred_len + future_gap))
        pred_time_end = pred_time_start + pd.Timedelta(hours=(future_len - 1))
        if pred_time_end > frame.index[-1]:
            break
        x.append(frame.loc[time: time + pd.Timedelta(hours=(pred_len - 1))].to_numpy())
        y.append(frame.loc[pred_time_start: pred_time_end].to_numpy())
        pred_times.append(pred_time_start)

    return x, y
    

def get_data(filename, source, valid_set=True):
    """
    加载数据
    :param filename: 存放数据的文件名
    :param source: 数据来源
    :param valid_set: 是否生成验证集
    """

    if source == 'lihui':
        x, y = get_lihui_data(filename)
    elif source == 'nanjing':
        x, y = get_nanjing_data(filename)
    elif source == 'beihang':
        x, y, z = get_beihang_data(filename)
    else:
        raise ValueError('No such source: ' + source)

    x = data_process.section_normalization(np.array(x))
    y, normal_y = data_process.section_normalization_with_normalizer(np.array(y))
    z = np.array(z)

    if valid_set:
        train_loc = int(0.7 * len(x))
        val_loc = int(0.8 * len(x))
        # test_loc = int(0.9 * len(x))
        x_train, y_train, feature_train = x[:train_loc], y[:train_loc], z[:train_loc]
        # x_train, y_train = np.concatenate((x[:train_loc], x[test_loc:])), np.concatenate((y[:train_loc], y[test_loc:]))
        x_val, y_val, feature_val = x[train_loc: val_loc], y[train_loc: val_loc], z[train_loc:val_loc]
        x_test, y_test, feature_test = x[val_loc: ], y[val_loc: ], z[val_loc:]
        # x_val, x_val1, y_val, y_val1 = train_test_split(x_test.copy(), y_test.copy(), test_size=0.66)

        print(f'数据集规模: x_train: {x_train.shape}, y_train: {y_train.shape},',
              f'x_val: {x_val.shape}, y_val: {y_val.shape},',
              f'x_test: {x_test.shape}, y_test: {y_test.shape}',
              f'feature_train: {feature_train.shape},feature_val: {feature_val.shape},feature_test: {feature_test.shape}'
              )
        return x_train, y_train, x_val, y_val, x_test, y_test, normal_y, feature_train, feature_test, feature_val
    else:
        train_loc = int(0.2 * len(x))
        test_loc = int(0.5 * len(x))
        x_train, y_train = np.concatenate((x[:train_loc], x[test_loc:])), np.concatenate((y[:train_loc], y[test_loc:]))
        x_test, y_test = x[train_loc:test_loc], y[train_loc:test_loc]
        return x_train, y_train, x_test, y_test, normal_y
