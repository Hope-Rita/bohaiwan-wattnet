import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils.util import convert_to_gpu, save_model
from utils.calculate_metric import calculate_metrics, normalized_transform
from utils.load_config import get_Parameter
from tqdm import tqdm
import copy
import numpy as np
import os


def train_model(model: nn.Module, data_loader, loss_func: callable, optimizer, num_epochs, model_folder,
                tensorboard_folder: str, teaching_force=False, **kwargs):
    phases = ['train', 'val', 'test']
    writer = SummaryWriter(tensorboard_folder)
    model = convert_to_gpu(model)
    loss_func = convert_to_gpu(loss_func)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=8, threshold=1e-4, min_lr=1e-6)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1], gamma=0.1)
    save_dict = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}
    loss_global = 100000
    list_loss = []
    global_step = num_epochs * get_Parameter('batch_size')
    for epoch in range(num_epochs):
        running_loss = {phase: 0.0 for phase in phases}
        for phase in phases:
            if phase == 'train':
                model.train()
                kwargs['is_eval'] = False
            else:
                kwargs['is_eval'] = True
                model.eval()
            steps, predictions, targets = 0, list(), list()
            tqdm_loaders = tqdm(enumerate(data_loader[phase]))
            for step, (features, truth, covariate) in tqdm_loaders:
                features = convert_to_gpu(features)
                truth = convert_to_gpu(truth)
                covariate = convert_to_gpu(covariate)
                if kwargs['pre_train']:
                    truth = features

                # global_step = (epoch) * 250 + step / 100

                kwargs['global_step'] = global_step
                with torch.set_grad_enabled(phase == 'train'):
                    if kwargs['return_attn']:
                        outputs, attn_list = model(features, truth, kwargs)
                    else:
                        outputs = model(features)

                    if get_Parameter('loss_normalized'):
                        outputs = outputs.detach().cpu().numpy()
                        truth = truth.detach().cpu().numpy()
                        outputs, truth = normalized_transform(outputs, truth, **kwargs)
                        outputs = torch.from_numpy(outputs).requires_grad_(requires_grad=True)
                        truth = torch.from_numpy(truth).requires_grad_(requires_grad=True)

                    # attn = [i for i in np.arange(6,6*(get_Parameter('target')) + 6, 6)]
                    # attn = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
                    loss = loss_func(truth[:,0,:], outputs[:,0,:])
                    # print(truth.shape)
                    if get_Parameter('target') > 1:
                        for time in range(1, get_Parameter('target')):
                            loss += loss_func(truth[:,time,:], outputs[:,time ,:])
                    # loss = loss_func(truth, outputs)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        list_loss.append(loss.item())
                        global_step += 1
                targets.append(truth.detach().cpu().numpy())
                with torch.no_grad():
                    predictions.append(outputs.cpu().numpy())
                running_loss[phase] += loss.item()
                steps += truth.size(0)

                tqdm_loaders.set_description(f'{phase} epoch:{epoch}, {phase} loss: {running_loss[phase]/steps}')

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)
            if phase == 'train':
                np.save('train.npy',predictions)
                np.save('target_train.npy',targets)
                np.save('loss.npy',list_loss)
            scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1),
                                       targets.reshape(targets.shape[0], -1), mode='train', **kwargs)
            print(scores)
            writer.add_scalars(f'score/{phase}', scores, global_step=epoch)
            if phase == 'val' and scores['RMSE'] < loss_global:
                loss_global = scores['RMSE']
                save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()), epoch=epoch,
                                 optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                save_model(f'{model_folder}/best_model_2-12.pkl', **save_dict)

        scheduler.step(running_loss['train'])
        writer.add_scalars('Loss', {
            f'{phase} loss': running_loss[phase] for phase in phases
        }, global_step=epoch)

    # save_model(f'{model_folder}/best_model_16.pkl', **save_dict)
    model.load_state_dict(save_dict['model_state_dict'])
    return model
