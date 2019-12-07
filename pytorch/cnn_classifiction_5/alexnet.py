# encoding: utf-8

"""
@author: sunxianpeng
@file: alexnet_offical.py
@time: 2019/11/18 12:54
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self,num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
        # (150 - 11 + 2 * 2)/ 4 + 1 = 36
            nn.Conv2d(3,64,11,4,padding=2),
            # inplace=True，是对于Conv2d这样的上层网络传递下来的tensor直接进行修改，
            # 好处就是可以节省运算内存，不用多储存变量
            nn.ReLU(inplace=True),
        # (36 - 3 + 2 * 0)/ 2 + 1 = 17
            nn.MaxPool2d(kernel_size=3,stride=2),
        # (17 - 5 + 2 * 2)/ 1 + 1 = 17
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            # (17 - 3 + 2 * 0)/ 2 + 1 = 8
            nn.MaxPool2d(kernel_size=3,stride=2),
            # (8 - 3 + 2 * 1)/ 1 + 1 = 8
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            # (8 - 3 + 2 * 1)/ 1 + 1 = 8
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            # (8 - 3 + 2 * 1)/ 1 + 1 = 8
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            # (8 - 3 + 2 * 0)/ 1 + 1 = 6
            nn.MaxPool2d(kernel_size=3,stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        print(x)
        result = x.max(dim=1)
        print(result)
        return x


if __name__ == '__main__':
    size = 150
    x = torch.randn(1, 3, size, size)
    model = AlexNet(num_classes=10)
    model(x)