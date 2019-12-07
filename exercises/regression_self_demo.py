# encoding: utf-8

"""
@author: sunxianpeng
@file: regression_self_demo.py
@time: 2019/11/14 18:43
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

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


class Net(torch.nn.Module):
    def __init__(self,input_size,num_classes):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size,10,bias=True)
        self.linear2 = nn.Linear(10,20,bias=True)
        self.linear3 = nn.Linear(20,num_classes,bias=True)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


if __name__ == '__main__':
    batch_size = 5
    input_size = 3
    num_classes = 2
    epochs = 100000

    inputs, targets = get_data()
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    train_ds = TensorDataset(inputs,targets)
    train_d1 = DataLoader(train_ds,batch_size,shuffle=True)

    # model = nn.Linear(3,2)#
    model = Net(input_size,num_classes)
    loss_fn = nn.functional.mse_loss
    opt = optimizer.SGD(model.parameters(),momentum=0.9,lr=0.000001)
    # opt = torch.optim.SGD(model.parameters(), lr=10**-5)
    for epoch in range(epochs):
        for x,y in train_d1:
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if(epoch+1) % 10000 == 0:
            print('Epoch [{}/{}], Loss:{:0.4f}'.format(epoch+1,epochs,loss.item()))

    test = torch.from_numpy(np.array([73,67,43],dtype='float32'))
    # [56, 70]
    print(model(test))
