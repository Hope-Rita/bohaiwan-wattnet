from utils.config import Config

config_path = 'config.json'
conf = Config(config_path)

import time
from dataset import get_data
from wattnet_predict import wattnet_predict
from utils import data_process
from utils import draw_pic
from utils import metric

pred_res_dir = None

if __name__ == '__main__':
    pred_target_filename = conf.predict_target
    sensor_name = conf.get_config('data-parameters', 'valid-sensors')
    x_train, y_train, x_test, y_test, normal_y = get_data(pred_target_filename)
    pred = wattnet_predict(x_train, y_train, x_test)
    pred = data_process.reverse_section_normalization(pred, normal_y)
    y_test = data_process.reverse_section_normalization(y_test, normal_y)

    # 输出预测指标
    metric.metric_for_each_sensor(y_test, pred, sensor_name)
    # 绘制对比图
    for i in range(pred.shape[2]):
        save_path = {'dir': conf.get_config('predict-pics', 'local' if conf.run_on_local else 'server'),
                     'filename': time.strftime('%Y-%m-%d-%H-%M-%S') + f'_sensor-{sensor_name[i]}.png'
                     }
        draw_pic.compare(y_test[:, 0, i], pred[:, 0, i], save_path=save_path, title_info=f'sensor-{sensor_name[i]}')
    print('Process end.')
