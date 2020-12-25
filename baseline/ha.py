import numpy as np


def ha_predict(x_train, y_train, x_test):
    return np.mean(x_test, axis=1)
