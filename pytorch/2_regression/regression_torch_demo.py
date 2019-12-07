# encoding: utf-8

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
def get_data():
    inputs = np.array([[73,67,43],
                       [91,88,64],
                       [87,134,58],
                       [102,43,37],
                       [69,96,70]],dtype='float32')
    targets = np.array([[56,70],
                        [81,101],
                        [119,133],
                        [22,37],
                        [103,119]],dtype='float32')
    return inputs,targets

def fit(train_d1,num_epochs, model, loss_fn,opt):
    # 迭代num_epochs次
    for epoch in range(num_epochs):
        # 使用分批数据进行训练
        for xb,yb in train_d1:
            #训练模型
            pred = model(xb)
            # mse损失函数
            loss = loss_fn(pred,yb)
            # 计算梯度，反向传播
            loss.backward()
            # 使用梯度更新参数
            opt.step()
            # 重置梯度为 0
            opt.zero_grad()
        if(epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss:{:0.4f}'.format(epoch+1,num_epochs,loss.item()))


if __name__ == '__main__':
    inputs, targets = get_data()
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    # 接下来我们创建一个TensorDataset和一个DataLoader：
    # 将特征和对应标签合并在一起
    train_ds = TensorDataset(inputs,targets)
    # print(train_ds[:3])
    batch_size = 5
    train_d1 = DataLoader(train_ds,batch_size,shuffle=True)
    # 定义一个三个输入维度两个输出维度的线性函数模型,
    # 使用nn.linear自动完成初始化工作。
    model = nn.Linear(3,2)
    # print(model.weight)
    # print(model.bias)
    # print(model.parameters())
    # 用内置损失函数mse_loss
    loss_fn = F.mse_loss
    # loss = loss_fn(model(inputs),targets)
    # 优化的时候，我们可以使用优化器optim.SGD，不用手动操作模型的权重和偏差。
    opt = torch.optim.SGD(model.parameters(), lr=10**-5)
    # 训练模型
    fit(train_d1,100,model,loss_fn,opt)
    preds = model(inputs)
    print(preds)