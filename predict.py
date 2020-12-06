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
    # 存放预测结果文件的路径
    pred_target_filename = conf.predict_target
    x_train, y_train, x_test, y_test, normal_y = get_data(pred_target_filename)
    pred = wattnet_predict(x_train, y_train, x_test)
    pred = data_process.reverse_section_normalization(pred, normal_y)
    y_test = data_process.reverse_section_normalization(y_test, normal_y)

    # 输出预测指标
    metric.metric_for_each_sensor(y_test, pred)
    # 绘制对比图
    for i in range(pred.shape[2]):
        save_path = {'dir': conf.get_config('predict-pics', 'local' if conf.run_on_local else 'server'),
                     'filename': time.strftime('%H:%M:%S') + f'_sensor-{i + 1}'
                     }
        draw_pic.compare(y_test[:, 0, i], pred[:, 0, i], save_path=save_path, title_info=f'sensor-{i + 1}')
    print('Process end.')
