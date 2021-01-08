import numpy as np
from sklearn.linear_model import LinearRegression
from utils.config import Config


conf = Config()


def lr_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def lr_multiple_predict(x_train, y_train, x_test):
    m, p, s = x_train.shape
    x_train = x_train.reshape(m, p * s)
    y_train = np.mean(y_train, axis=1)
    linear_reg = model_fit(x_train, y_train)
    pred = predict(linear_reg, x_test.reshape(len(x_test), p * s))
    future_len = conf.get_config('data-parameters', 'future-len')
    return np.expand_dims(pred, 1).repeat(future_len, axis=1)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(x_train, y_train)
    return linear_reg
