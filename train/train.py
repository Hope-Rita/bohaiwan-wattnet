import torch
from torch import optim

from baseline.GRU import GRU
from baseline.LSTM import LSTM
from baseline.RNN import RNN
from models.optim.NoamOpt import NoamOpt
from models.wattnet import WATTNet
from utils.data_loader import load_data
from utils.load_config import get_Parameter
from utils.save import xw_toExcel
from utils.util import convert_to_gpu, build_sparse_matrix
from utils.calculate_metric import calculate_metrics, normalized_transform
from train.train_model import train_model
from models.optim.BERTAdam import BERTAdam
import os
import numpy as np
from tqdm import tqdm
import shutil



def create_model(model_name, params):

    if model_name.startswith('WATTNet'):
        pred_len = get_Parameter((model_name,'series_len'))
        sensor_num = get_Parameter((model_name,'in_dim'))
        future_len = get_Parameter((model_name,'out_dim'))
        depth = get_Parameter((model_name,'depth'))
        n_repeat = get_Parameter((model_name,'n_repeat'))
        show_attn = get_Parameter((model_name,'show_attn'))
        embed_dim = get_Parameter((model_name,'embed_dim'))
        feat_dim = get_Parameter('covariate_size')
        return WATTNet(series_len=pred_len, in_dim=sensor_num, emb_dim=embed_dim, out_dim=future_len, depth=depth, n_repeat=n_repeat,feat_dim=feat_dim, show_attn_alpha=show_attn)

    elif model_name == 'lstm':
        input_size = get_Parameter('input_size')
        hidden_size = get_Parameter((model_name, 'hidden_size'))
        num_layers = get_Parameter((model_name, 'num_layers'))
        output_size = get_Parameter('target')
        return LSTM(input_size, hidden=hidden_size, num_layers=num_layers, output_len=output_size)
    elif model_name == 'rnn':
        input_size = get_Parameter('input_size')
        hidden_size = get_Parameter((model_name, 'hidden_size'))
        num_layers = get_Parameter((model_name, 'num_layers'))
        output_size = get_Parameter('target')
        return RNN(input_size, hidden=hidden_size, num_layers=num_layers, output_len=output_size)
    elif model_name == 'gru':
        input_size = get_Parameter('input_size')
        hidden_size = get_Parameter((model_name, 'hidden_size'))
        num_layers = get_Parameter((model_name, 'num_layers'))
        output_size = get_Parameter('target')
        return GRU(input_size, hidden=hidden_size, num_layers=num_layers, output_len=output_size)

def test_model(model, data_loader, mode, teaching_force=False, **kwargs):
    predictions = list()
    targets = list()
    tqdm_loader = tqdm(enumerate(data_loader))
    model = convert_to_gpu(model)
    if kwargs['return_attn']:
        attn_record = list()
    with torch.no_grad():
        kwargs['is_eval'] = True
        for step, (features, truth, covariate) in tqdm(tqdm_loader):
            features = convert_to_gpu(features)
            truth = convert_to_gpu(truth)
            covariate = convert_to_gpu(covariate)
            if kwargs['pre_train']:
                truth = features
            if kwargs['return_attn']:
                outputs, attn_list = model(features, truth, **kwargs)
                attn_record.append(attn_list.cpu().numpy())
            else:
                outputs = model(features,covariate)
            outputs = outputs.detach().cpu().numpy()
            truth = truth.detach().cpu().numpy()
            outputs, truth = normalized_transform(outputs, truth, **kwargs)
            targets.append(truth)
            predictions.append(outputs)
    pre2 = np.concatenate(predictions)
    tar2 = np.concatenate(targets)
    if isinstance(kwargs['pca'], np.ndarray):
        pre2 = np.matmul(pre2.reshape(pre2.shape[0], -1), kwargs['pca'])
        tar2 = np.matmul(tar2.reshape(tar2.shape[0], -1), kwargs['pca'])
    print(pre2.shape)
    data = calculate_metrics(pre2, tar2, mode, **kwargs)
    print(data)

    xw_toExcel(data, 'result/'+get_Parameter('model_name')+'/'+get_Parameter('model_name')+'-result.xlsx')
    np.save('result/'+get_Parameter('model_name')+'/'+get_Parameter('model_name')+'-pred.npy',pre2)
    np.save('result/'+get_Parameter('model_name')+'/'+get_Parameter('model_name')+'-truth.npy',tar2)
    if kwargs['return_attn']:
        return attn_record




def params_setting(scaler):
    params = dict()
    params['scaler'] = scaler
    # params['attn'] = False
    params['embedding'] = True
    params['require_embedding'] = False
    params['return_attn'] = False
    params['pre_train'] = False
    params['load_pretrain'] = False
    params['pca'] = False
    return params


def load_pre_train(model_name, model, pretrain_model_path):
    load_path = '/home/zoutao/SHM-prediction/save_models/DCRNN/best_model_2-5.pkl'
    pretrain_model = torch.load(os.path.join(pretrain_model_path, load_path))['model_state_dict']
    # print('pretrain model dict {}'.format(pretrain_model))
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_model.items() if k in model_dict}
    print(pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model


def train():
    batch_size = get_Parameter('batch_size')
    model_name = get_Parameter('model_name')
    data_path = get_Parameter('data_path')

    data_loader, scaler = load_data(data_path, batch_size, normalized=get_Parameter('normalized'))
    params = params_setting(scaler)

    model = create_model(model_name, params)
    param_num = 0
    for name, param in model.named_parameters():
        print(name, ':', param.size())
        param_num = param_num + np.product(np.array(list(param.size())))
    print('param number:' + str(param_num))

    model_folder = f"save_models/{model_name}"
    tensorboard_folder = f'runs/{model_name}'

    # 训练
    if get_Parameter('mode') == 'train':
        num_epoches = get_Parameter('epochs')
        os.makedirs(model_folder, exist_ok=True)
        shutil.rmtree(tensorboard_folder, ignore_errors=True)
        os.makedirs(tensorboard_folder, exist_ok=True)
        loss = torch.nn.MSELoss()

        if get_Parameter('optim') == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=get_Parameter((model_name, 'lr')),
                                   weight_decay=get_Parameter('weight_decay'))
        elif get_Parameter('optim') == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=get_Parameter((model_name, 'lr')), momentum=0.9)
        elif get_Parameter('optim') == 'RMSProp':
            optimizer = optim.RMSprop(model.parameters(), lr=get_Parameter((model_name, 'lr')))
        elif get_Parameter('optim') == 'BERTAdam':
            optimizer = BERTAdam(model.parameters(), lr=get_Parameter((model_name, "lr")), warmup=0.1)
        elif get_Parameter('optim') == 'NoamOpt':
            optimizer = NoamOpt(model_size=256, factor=1, warmup=400,
                                optimizer=optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9))
        else:
            raise NotImplementedError

        if params['load_pretrain']:
            model = load_pre_train(model_name, model, "")
        if get_Parameter('model_name')!='MyModel':
            model = train_model(model, data_loader=data_loader, loss_func=loss, optimizer=optimizer,
                                num_epochs=num_epoches, model_folder=model_folder,
                                tensorboard_folder=tensorboard_folder, **params)
        # else:
        #     model = train_my_model(model, data_loader=data_loader, loss_func=loss, optimizer=optimizer,
        #                         num_epochs=num_epoches, model_folder=model_folder,
        #                         tensorboard_folder=tensorboard_folder, **params)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model_24-15.pkl'))['model_state_dict'])
    attn = test_model(model, data_loader['test'], mode='test', **params)
    return attn
