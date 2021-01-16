from utils.config import Config

config_path = 'config.json'
conf = Config(config_path)

from dataset import get_data
from utils import data_process
from utils import metric
from baseline.lr import lr_single_predict
from baseline.svr import svr_single_predict
from baseline.mlp import mlp_single_predict
from baseline.ha import ha_predict
from baseline.recurrent import rnn_single_predict, gru_single_predict, lstm_single_predict


if __name__ == '__main__':
    pred_target_filename = conf.predict_target
    sensor_name = conf.get_config('data-parameters', 'valid-sensors')
    pred_func = rnn_single_predict

    x_train, y_train, x_test, y_test, normal_y = get_data(pred_target_filename, 'nanjing', valid_set=False)

    m = []
    for i in range(len(sensor_name)):
        pred = pred_func(x_train=x_train[:, :, i], y_train=y_train[:, :, i], x_test=x_test[:, :, i])
        pred = normal_y[i].inverse_transform(pred)
        y_test[:, :, i] = normal_y[i].inverse_transform(y_test[:, :, i])
        metrics = metric.all_metric(y_test[:, :, i].reshape(-1), pred.reshape(-1))
        m.append(metrics)
        print(f'sensor-{sensor_name[i]}: {metrics}')
    data_process.average_metric(m, data_process.avg)

    print('Process end.')
