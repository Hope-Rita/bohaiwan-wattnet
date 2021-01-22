from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from utils.config import Config


conf = Config()
hidden_size = tuple(conf.get_config('mlp-hyper-para', 'hidden'))


def mlp_multiple_predict(x_train, y_train, x_test):
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
    reg = model_fit(x_train, y_train)
    pred = predict(reg, x_test.reshape(len(x_test), p * s))
    return pred.reshape(-1, future_len, s)


def mlp_pca_predict(x_train, y_train, x_test):
    """
    :param x_train: m, pred_len, sensor_num
    :param y_train: m, future_len, sensor_num
    :param x_test: m_test, pred_len, sensor_num
    :return: pred in shape(m_test, future_len, sensor_num)
    """
    m, p, s = x_train.shape
    future_len = conf.get_config('data-parameters', 'future-len')
    x_train = x_train.reshape(m, p * s)
    x_test = x_test.reshape(len(x_test), p * s)
    y_train = y_train.reshape(m, future_len * s)

    estimator = PCA(n_components=36)
    x_train_pca = estimator.fit_transform(x_train)
    x_test_pca = estimator.transform(x_test)

    reg = model_fit(x_train_pca, y_train)
    pred = predict(reg, x_test_pca)
    return pred.reshape(-1, future_len, s)


def mlp_single_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=hidden_size, activation='identity')
    model.fit(x_train, y_train)
    return model
