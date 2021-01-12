from utils.config import Config

config_path = 'config.json'
conf = Config(config_path)

from dataset import get_data
from utils import data_process
from utils import metric
from baseline.lr import lr_predict
from baseline.svr import svr_predict
from baseline.mlp import mlp_predict
from baseline.ha import ha_predict
from baseline.recurrent import rnn_union_predict, lstm_union_predict, gru_union_predict


if __name__ == '__main__':
    pred_target_filename = conf.predict_target
    sensor_name = conf.get_config('data-parameters', 'valid-sensors')
    pred_func = lr_predict

    x_train, y_train, x_test, y_test, normal_y = get_data(pred_target_filename, 'nanjing', valid_set=False)
    y_train = y_train[:, -1, :]
    y_test = y_test[:, -1, :]
    y_test = data_process.reverse_col_transform(y_test, normal_y)

    for i in range(len(sensor_name)):
        pred = pred_func(x_train=x_train[:, :, i], y_train=y_train[:, i].reshape(-1), x_test=x_test[:, :, i])
        pred = normal_y[i].inverse_transform(pred)
        metrics = metric.all_metric(y_test[:, i], pred)
        print(f'sensor-{sensor_name[i]}: {metrics}')

    print('Process end.')
