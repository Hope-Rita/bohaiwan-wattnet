from sklearn.linear_model import LinearRegression
from utils.config import Config


conf = Config()


def lr_single_predict(x_train, y_train, x_test):
    """
    :param x_train: m, pred_len
    :param y_train: m, future_len
    :param x_test: m_test, pred_len
    :return: pred in shape(m_test, future_len)
    """
    linear_reg = model_fit(x_train, y_train)
    pred = predict(linear_reg, x_test)
    return pred


def lr_multiple_predict(x_train, y_train, x_test):
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
    linear_reg = model_fit(x_train, y_train)
    pred = predict(linear_reg, x_test.reshape(len(x_test), p * s))
    return pred.reshape(-1, future_len, s)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    return linear_reg
