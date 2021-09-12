import sys

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

from utils.calculate_metric import calculate_metrics, masked_mape_np
from utils.data_generator import Data_utility
from utils.data_loader import load_data
from utils.load_config import get_Parameter
from utils.save import xw_toExcel
from utils.util import convert_to_gpu

filename = '/home/zoutao/lwt-prediction/data/data.csv'
d = Data_utility(data_path=filename, train=0.7, valid=0.1, window=30, target=15)
window = 30
test = np.concatenate(d.test, axis=1)
train = np.concatenate(d.train, axis=1)
train = train[:, :, :get_Parameter('input_size')]
test = test[:, :, :get_Parameter('input_size')]

train = np.transpose(train, (0, 2, 1))
train = np.reshape(train, [-1, train.shape[-1]])
# permutation = np.random.permutation(train.shape[0])
# train = train[permutation, :]
# permutation = np.random.permutation(test.shape[0])
# test = test[permutation, :]
x, y = np.split(train, [window], axis=1)
y= y[:,0]
# print(x.shape)
# print(y.shape)
# print(x[0,:])
#
# model = xgb.XGBRegressor(learning_rate=0.8, max_depth=35, gamma=1e-3, n_estimators=10, objective='reg:squarederror')
# cv_params = {'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}
# cv_params = {'subsample': [0.9, 0.95, 1.0], 'colsample_bytree': [0.9, 0.95, 1.0]}
# cv_params = {'learning_rate': [0.09, 0.07, 0.075, 0.08]}
# other_params = {'learning_rate': 0.08, 'n_estimators': 400, 'max_depth': 3, 'min_child_weight': 2, 'seed': 0,
#                     'subsample': 0.95, 'colsample_bytree': 1.0, 'gamma': 1e-5, 'reg_alpha': 0.05, 'reg_lambda': 1}
#
other_params = {'learning_rate': 0.99, 'n_estimators': 24, 'max_depth': 32, 'min_child_weight': 4.0, 'seed': 0,
                    'subsample': 1, 'colsample_bytree': 1.0, 'gamma': 0.11, 'reg_alpha': 0.85, 'reg_lambda': 0.605, 'objective':'reg:squarederror'}
model = xgb.XGBRegressor(**other_params)

# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=5)
# optimized_GBM.fit(x,y)
# evalute_result = optimized_GBM.scoring
# print('每轮迭代运行结果:{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


model.fit(x, y)
# test = np.transpose(test, (0, 2, 1))
# test = np.reshape(test, [-1, test.shape[-1]])

rmses, maes, mapes, pccs = list(), list(), list(), list()
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

np.save('../result/xgboost/truth-xgboost.npy', y_test)
np.save('../result/xgboost/pred-xgboost.npy', pred)
data = calculate_metrics(pred, y_test, mode='test')

xw_toExcel(data,'../result/xgboost/xgboost-result.xlsx')
print(data)