# encoding: utf-8

"""
@author: sunxianpeng
@file: lenet.py
@time: 2019/11/18 11:58
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset


from torchvision import transforms
import math

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(1,6,5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(6,16,5),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
        nn.Linear(16*5*5,120),
        nn.ReLU(),
        nn.Linear(120,84),
        nn.ReLU(),
        nn.Linear(84,10)
        )

    def forward(self,x):
        x = self.features(x)
        print(x.size())#torch.Size([1, 16, 5, 5])
        print(x.shape)#torch.Size([1, 16, 5, 5])
        x = x.view(x.size(0),-1)#torch.Size([1, 400])
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 代入数据检验
    y = torch.randn(1, 1, 32, 32)
    model = LeNet()
    model(y)