# encoding: utf-8

import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from torch.functional import F

from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

def split_indices(total_num,v_percent):
    v_num = int(total_num * v_percent)
    # permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
    indexs = np.random.permutation(total_num)
    return indexs[v_num:2*v_num],indexs[3*v_num:4*v_num]

class MnistModel(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size,num_classes)

    def forward(self, x):
        x = x.reshape(-1,28 * 28)
        out = self.linear(x)
        return out



if __name__ == '__main__':
    datasets =MNIST("./mnist",train=True,transform=transforms.ToTensor(),download=True)
    # data
    t_indices,v_indices = split_indices(len(datasets),0.2)
    batch_size = 100
    t_sampler = SubsetRandomSampler(t_indices)
    v_sampler = SubsetRandomSampler(v_indices)
    t_loader = DataLoader(datasets,batch_size,sampler=t_sampler)
    v_loader = DataLoader(datasets,batch_size,sampler=v_sampler)

    in_size = 28 * 28
    num_classes = 10
    epochs = 5
    modle = MnistModel(in_size,num_classes)
    loss_fn = F.cross_entropy
    lr = 10**-3
    opt = torch.optim.SGD(modle.parameters(),lr = lr)
#     train
    for epoch in range(epochs):

        for x,y in t_loader:
            pred = modle(x)
            loss = loss_fn(pred,y)
            opt.zero_grad()
            loss.backward()
            opt.step()

#     valdation
    with torch.no_grad():
        total_acc = 0.
        total_loss = 0.
        for x,y in v_loader:
            pred = modle(x)
            loss = loss_fn(pred,y)















