from sklearn.neural_network import MLPRegressor
from utils.config import Config


conf = Config()
hidden_size = tuple(conf.get_config('mlp-hyper-para', 'hidden'))


def mlp_multiple_predict(x_train, y_train, x_test):
    m, p, s = x_train.shape
    future_len = conf.get_config('data-parameters', 'future-len')
    x_train = x_train.reshape(m, p * s)
    y_train = y_train.reshape(m, future_len * s)
    reg = model_fit(x_train, y_train)
    pred = predict(reg, x_test.reshape(len(x_test), p * s))
    return pred.reshape(-1, future_len, s)


def mlp_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=hidden_size, activation='identity')
    model.fit(x_train, y_train)
    return model
