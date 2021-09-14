import numpy as np
import matplotlib.pyplot as plt
#
import torch
# load_path = '/home/zoutao/SHM-prediction/save_models/DCRNN/best_model_2-12.pkl'
# pretrain_model = torch.load(load_path)['model_state_dict']
# import torch.nn.functional as F
# # print(pretrain_model)
# nodevec1_2 = pretrain_model['nodevec1']
# nodevec2_2 = pretrain_model['nodevec2']
# static = torch.tensor(np.load('/home/zoutao/SHM-prediction/data/adjcent.npy').astype('float32'))
# adp = F.softmax(F.relu(torch.mm(nodevec1_2, nodevec2_2)), dim=1)
# # adp = F.relu(torch.tanh(2.0 * torch.mm(nodevec1_2, nodevec2_2)))
# # x = np.load('adjacent.npy')
# print(nodevec1_2,nodevec2_2)
# x = adp.cpu()
# # print(x)
# plt.imshow(x, cmap="Blues", vmin=0, vmax=1)
# plt.colorbar()
# name = "delayer0"
# # plt.savefig(name +".png")
# print(F.softmax(static,dim=1))
# plt.show()

#可视化结果
from sklearn.metrics import mean_squared_error
from utils.load_config import get_Parameter

data = np.load('result/'+get_Parameter('model_name')+'/'+get_Parameter('model_name')+'-truth.npy')
print(data.shape)
time = 6
data1 = data[:,time,:]
# data2 = data[:,4,:]
# data3 = data[:,2,:]

pred = np.load('result/'+get_Parameter('model_name')+'/'+get_Parameter('model_name')+'-pred.npy')
pred1 =pred[:,time,:]
# pred2 = pred[:,11,:]
# truth3 = truth[:,2,:]

# print(pred[:,10,3]-pred[:,8,3])

index = 2
start = 0
len = min(pred1.shape[0],data1.shape[0])
# len -= 11
# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
# len = 100
y = range(len)
plt.plot(y, data1[start:start+len,index],  '-', color='red', alpha=0.8, linewidth=1.5,label='truth')
plt.plot(y, pred1[start:start+len,index],'-', color='blue', alpha=0.8, linewidth=0.8,label='pred')
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('x-'+str(time)+'-'+str(index))
plt.ylabel('y-'+str(time)+'-'+str(index))
# pre = 0
# for data in truth1[:, index]:
#     if pre == data:
#         print(pre)
#     pre = data
print(data1[:,index],pred1[:,index])
plt.show()
mse = np.sqrt(mean_squared_error(data1[:,index], pred1[:,index]))
# print(data1[:,index],truth1[:,index])
print(mse)

# loss = np.load('/home/zoutao/SHM-prediction/loss.npy')
# len = 2500
# y = range(len)
# plt.plot(y, loss[-len:],  '-', color='red', alpha=0.8, linewidth=1.5,label='truth')
#
# plt.savefig('loss.png')
# plt.show()