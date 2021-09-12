from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVR

from utils.calculate_metric import calculate_metrics, masked_mape_np
from utils.data_generator import Data_utility
import numpy as np

from utils.load_config import get_Parameter
from utils.save import xw_toExcel


def svr_predict(x_train, y_train, x_test):
    return model_fit(x_train, y_train).predict(x_test)


def predict(model, x_test):
    return model.predict(x_test)


def model_fit(x_train, y_train):
    svm_reg = LinearSVR()
    svm_reg.fit(x_train, y_train)
    return svm_reg

filename = '/home/zoutao/lwt-prediction/data/data.csv'
d = Data_utility(data_path=filename, train=0.7, valid=0.1, window=30, target=15)
window = 30
test = np.concatenate(d.test, axis=1)
train = np.concatenate(d.train, axis=1)
# print(test.shape)
train = train[:, :, :get_Parameter('input_size')]
test = test[:, :, :get_Parameter('input_size')]
print(train.shape,test.shape)

train = np.transpose(train, (0, 2, 1))
train = np.reshape(train, [-1, train.shape[-1]])
x_train, y_train = np.split(train, [window], axis=1)
y_train = y_train[:, 0]
model = model_fit(x_train,y_train)

print(x_train.shape,y_train.shape)

test = np.transpose(test, (0, 2, 1))
pred = []
for ts in test:
    ts = ts.reshape(-1, ts.shape[-1])
    test_inputs, labels = np.split(ts, [window], axis=1)
    labels = np.transpose(labels)[0]
    predictions = []
    for i in range(get_Parameter('target')):
        prediction = model.predict(test_inputs)
        prediction = prediction.reshape(prediction.shape[0],1)
        test_inputs = np.concatenate([test_inputs[:, 1:], prediction],axis=-1)
        predictions.append(prediction)

    predictions = np.concatenate(predictions,axis=-1)
    pred.append(predictions)

y_train, y_test = np.split(test, [window], axis=-1)
pred = np.concatenate(pred, axis=0)
print(pred.shape)
y_test = np.concatenate(y_test)
print(y_test.shape)
y_test = y_test.reshape(-1, get_Parameter('input_size'),y_test.shape[-1])
y_test= np.transpose(y_test, (0, 2, 1))
pred = pred.reshape(-1, get_Parameter('input_size'),get_Parameter('target'))
pred= np.transpose(pred, (0, 2, 1))
d.max = d.max[:4]
d.min = d.min[:4]
pred = pred * (d.max - d.min) + d.min
y_test = y_test * (d.max - d.min) + d.min

np.save('../result/svr/truth-svr.npy', y_test)
np.save('../result/svr/pred-svr.npy', pred)
data = calculate_metrics(pred, y_test, mode='test')

xw_toExcel(data,'../result/svr/svr-result.xlsx')
print(data)
