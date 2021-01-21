import numpy as np
import time
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
    pred_func = lr_single_predict
    exp_mode = 1

    x_train, y_train, x_test, y_test, normal_y = get_data(pred_target_filename, 'nanjing', valid_set=False)

    m = []
    pred_merge = np.zeros(y_test.shape)
    for i in range(len(sensor_name)):
        pred = pred_func(x_train=x_train[:, :, i], y_train=y_train[:, :, i], x_test=x_test[:, :, i])
        pred = normal_y[i].inverse_transform(pred)
        pred_merge[:, :, i] = pred
        y_test[:, :, i] = normal_y[i].inverse_transform(y_test[:, :, i])
        metrics = metric.all_metric(y_test[:, :, i].reshape(-1), pred.reshape(-1))
        m.append(metrics)
        print(f'sensor-{sensor_name[i]}: {metrics}')
    data_process.average_metric(m, data_process.avg)
    data_process.dump_pred_res(dirname=conf.get_config('predict-res-table', conf.run_location),
                               filename=time.strftime('%Y-%m-%d-%H-%M-%S') + '_' + pred_func.__name__ + '.csv',
                               y=y_test,
                               pred=pred_merge,
                               sensor_name=sensor_name
                               )
    print('Process end.')
