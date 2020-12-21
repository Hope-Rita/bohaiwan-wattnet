import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def all_predict(y_dict):

    for key in y_dict:
        y = y_dict[key]
        if key != 'True value':
            tmp = np.array([np.nan] * (len(y_dict['True value']) - len(y)))
            tmp[-1] = y_dict['True value'][len(y_dict['True value']) - len(y)]
            y = np.concatenate((tmp, y))

        plt.plot(y[200:], label=key)

    plt.legend()
    plt.show()


def compare(y, pred, save_path, title_info=None):
    plt.figure(figsize=(15, 8))
    plt.plot(pred, color='red', label='predict')
    plt.plot(y, label='truth')

    if title_info:
        plt.title(title_info)

    plt.legend()

    # 保存图像, 如果路径不存在则要新建
    if not os.path.exists(save_path['dir']):
        os.makedirs(save_path['dir'])
    plt.savefig(os.path.join(save_path['dir'], save_path['filename']))
    plt.clf()
    plt.close('all')


def draw_by_label(pic_path, pic_name, **items):
    plt.figure()
    for label in items:
        plt.plot(items[label], label=label)

    plt.title(pic_name)
    plt.legend()

    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    plt.savefig(os.path.join(pic_path, pic_name))

    plt.clf()
    plt.close('all')


def train_process_pic(train_loss, val_loss, title=None):
    plt.figure()
    plt.plot(train_loss, color='green', label='Train loss')
    plt.plot(val_loss, color='blue', label='Valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if title:
        plt.title(title)
    plt.show()


def show_attention_matrix(alpha, layer):
    """
    draw a heatmap using `alpha` matrix -- similarity matrix in self-attention modules
    :param alpha: shape (channel, channel)
    :param layer: layer-th layer in the model
    """
    channel_num = len(alpha)

    sns.set()
    heat_map = sns.heatmap(alpha,  cmap='YlGnBu')
    plt.title(f'layer-{layer}')
    plt.show()
