import numpy as np
# import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.load_config import get_Parameter
# from utils.util import record_predict_result, convert_to_gpu


def normalized_transform(predict, target, **kwargs):
    scaler = kwargs['scaler']
    if get_Parameter('normalized'):
        predict, target = scaler.inverse_transform(predict), scaler.inverse_transform(target)
    return predict, target


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
            # mask = labels>10
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        mse = np.nan_to_num(mse)
        return np.mean(mse)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            # labels = labels.astype('int32')
            mask = np.not_equal(labels, null_val)
            # mask = labels>10
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        mape = np.nan_to_num(mape)
        return np.mean(mape*100)


def cal_pcc(preds, labels):
    PCC = []
    for i in range(get_Parameter('input_size')):
        pcc, _ = pearsonr(preds[:,i],labels[:,i])
        PCC.append(pcc)
    return np.mean(PCC)

def get_metrics(predict, target):
    length_p, length_t = len(predict), len(target)
    if length_p != length_t:
        assert 'wrong ! cannot calculate metric'
    RMSEs, MAEs, MAPEs, PCCs = list(), list(), list(), list()
    for i in range(length_p):
        predict[i] = predict[i].reshape(predict[i].shape[0], -1)
        target[i] = target[i].reshape(target[i].shape[0], -1)
        mse = mean_squared_error(predict[i], target[i])
        mae = mean_absolute_error(predict[i], target[i])
        pcc= cal_pcc(predict[i], target[i])
        mape = masked_mape_np(predict[i], target[i], null_val=0)
        RMSEs.append(np.sqrt(mse))
        MAEs.append(mae)
        MAPEs.append(mape)
        PCCs.append(pcc)
    return RMSEs, MAEs, MAPEs, PCCs


def get_data_list(seperate_timestep, predict, target):
    predict_list, target_list = list(), list()
    for i in range(seperate_timestep):
        predict_list.append(predict[:, i, :])
        target_list.append(target[:, i, :])
    return predict_list, target_list


def calculate_metrics(predict, target, mode, **kwargs):
    if get_Parameter('normalized') and get_Parameter('loss_normalized'):
        predict, target = normalized_transform(predict, target, **kwargs)
    if mode == 'train':
        mse = mean_squared_error(predict, target)
        # mse = masked_mse_np(predict, target, null_val=0)
        mae = mean_absolute_error(predict, target)
        mape = masked_mape_np(predict, target, null_val=0)
        pcc, _ = pearsonr(predict.flatten(),target.flatten())
        return {
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'MAPE': mape,
            'PCC': pcc
        }
    seperate_timestep = get_Parameter('seperate_timestep')

    predict, target = get_data_list(seperate_timestep, predict, target)
    # target = [pickup_target, dropoff_target]
    print(len(predict))

    RMSEs, MAEs, MAPEs, PCCs = get_metrics(predict, target)
    return {
        'RMSE': RMSEs,
        'MAE': MAEs,
        'MAPE': MAPEs,
        'PCC': PCCs
    }
