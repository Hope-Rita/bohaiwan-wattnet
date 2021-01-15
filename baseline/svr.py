import numpy as np
from sklearn.svm import LinearSVR, SVR
from sklearn.multioutput import MultiOutputRegressor
from utils.config import Config


conf = Config()


def svr_multiple_predict(x_train, y_train, x_test):
    """
    :param x_train: m, pred_len, sensor_num
    :param y_train: m, future_len, sensor_num
    :param x_test: m_test, pred_len, sensor_num
    :return: pred in shape(m_test, future_len, sensor_num)
    """
    m, p, s = x_train.shape
    future_len = conf.get_config('data-parameters', 'future-len')
    x_train = x_train.reshape(m, p * s)
    y_train = y_train.reshape(m, future_len * s)

    reg = MultiOutputRegressor(SVR(kernel='linear'))
    reg.fit(x_train, y_train)
    pred = reg.predict(x_test.reshape(len(x_test), p * s))
    return pred.reshape(-1, future_len, s)


def svr_single_predict(x_train, y_train, x_test):
    """
    :param x_train: m, pred_len
    :param y_train: m, future_len
    :param x_test: m_test, pred_len
    :return: pred in shape(m_test, future_len)
    """
    reg = MultiOutputRegressor(SVR(kernel='linear'))
    reg.fit(x_train, y_train)
    return reg.predict(x_test)


