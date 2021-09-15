from utils.calculate_metric import calculate_metrics
from utils.config import Config
from utils.load_config import get_Parameter
from utils.save import xw_toExcel

config_path = 'config.json'
conf = Config(config_path)

import time
from dataset import get_data
from wattnet_predict import wattnet_predict
from utils import data_process
import numpy as np
from utils import draw_pic
from utils import metric


if __name__ == '__main__':
    pred_target_filename = conf.predict_target
    sensor_name = conf.get_config('data-parameters', 'valid-sensors')
    x_train, y_train, x_val, y_val, x_test, y_test, normal_y,feature_train,feature_test,feature_val = get_data(pred_target_filename, 'beihang')
    pred = wattnet_predict(x_train, y_train, x_val, y_val, x_test,feature_train, feature_test,feature_val)
    pred = data_process.reverse_section_normalization(pred, normal_y)
    y_test = data_process.reverse_section_normalization(y_test, normal_y)

    # # 输出预测指标
    # metrics = metric.metric_for_each_sensor(y_test, pred, sensor_name)
    # # 存储预测结果
    # now_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    # model_type = conf.get_config('model-config', 'save-name')
    # data_process.dump_csv(dirname='pred_res/metrics', filename=now_time + '_' + model_type + '.csv', data=metrics,
    #                       average_func=data_process.avg)
    # data_process.dump_pred_res(dirname=conf.get_config('predict-res-table', conf.run_location),
    #                            filename=now_time + '_' + model_type + '.csv',
    #                            y=y_test,
    #                            pred=pred,
    #                            sensor_name=sensor_name
    #                            )
    #
    # # 绘制对比图
    # for i in range(pred.shape[2]):
    #     save_path = {'dir': conf.get_config('predict-pics', conf.run_location),
    #                  'filename': f'{now_time}_{model_type}_sensor-{sensor_name[i]}.png'
    #                  }
    #     draw_pic.compare(y_test[:, 0, i], pred[:, 0, i], save_path=save_path, title_info=f'sensor-{sensor_name[i]}')
    data = calculate_metrics(pred, y_test, 'test')
    print(data)

    xw_toExcel(data, 'result/' + get_Parameter('model_name') + '/' + get_Parameter('model_name') + '-result.xlsx')
    np.save('result/' + get_Parameter('model_name') + '/' + get_Parameter('model_name') + '-pred.npy', pred)
    np.save('result/' + get_Parameter('model_name') + '/' + get_Parameter('model_name') + '-truth.npy', y_test)

    print('Process end.')
