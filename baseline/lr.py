import numpy as np
from sklearn.linear_model import LinearRegression
from utils.config import Config


conf = Config()


def lr_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def lr_multiple_predict(x_train, y_train, x_test):
    m, p, s = x_train.shape
    future_len = conf.get_config('data-parameters', 'future-len')
    x_train = x_train.reshape(m, p * s)
    # y_train = np.mean(y_train, axis=1)
    y_train = y_train.reshape(m, future_len * s)
    linear_reg = model_fit(x_train, y_train)
    pred = predict(linear_reg, x_test.reshape(len(x_test), p * s))
    # return np.expand_dims(pred, 1).repeat(future_len, axis=1)
    return pred.reshape(-1, future_len, s)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    return linear_reg
