# encoding: utf-8

"""
@author: sunxianpeng
@file: resnet.py
@time: 2019/11/28 16:18
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import numpy as np

def check_image_data(train_data):
    import matplotlib.pyplot as plt
    for im, label in train_data:
        print(im[0])
        print(label[0])
        img = im[0].permute(2,1,0)
        img = img.numpy()
        plt.imshow(img)
        plt.show()
        break

def data_tf(x):
    x =	x.resize((96,96),2)#将图片放大
    x =	np.array(x,	dtype='float32')/ 255 #
    x =	(x - 0.5)/ 0.5#标准化
    x =	x.transpose((2,0,1))#将channel放在第一位置
    x =	torch.from_numpy(x)
    return x

def conv3x3(in_channel,	out_channel,stride=1):
    return nn.Conv2d(in_channel,out_channel,3,stride=stride,padding=1,bias=False)

class residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)

class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        """
        :param in_channel: 输入通道数
        :param num_classes: 分类个数
        :param verbose:是否打印每次卷积后的数据shape
        """
        super(resnet, self).__init__()
        self.verbose = verbose
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block	1	output:	{}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block	2	output:	{}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block	3	output:	{}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block	4	output:	{}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block	5	output:	{}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    print("输入输出相同形状==============")
    test_net = residual_block(32, 32)
    test_x = torch.zeros(1, 32, 96, 96)
    print('input:	{}'.format(test_x.shape))
    test_y = test_net(test_x)
    print('output:	{}'.format(test_y.shape))

    print("输入输出形状不同==============")
    test_net = residual_block(3, 32, False)
    test_x = torch.zeros(1, 3, 96, 96)
    print('input:	{}'.format(test_x.shape))
    test_y = test_net(test_x)
    print('output:	{}'.format(test_y.shape))

    print("输入一下每个block之后的大小=======")
    test_net = resnet(3, 10, True)
    test_x = torch.zeros(1, 3, 96, 96)
    test_y = test_net(test_x)
    print('output:	{}'.format(test_y.shape))

    print("训练数据===============")
    from pytorch.cnn_classifiction_5.utils import train
    data_path = './cifar10'
    train_set = CIFAR10(data_path, train=True, transform=data_tf)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10(data_path, train=False, transform=data_tf)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    check_image_data(train_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet(3, 10)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    train(net, train_data, test_data, 20, optimizer, criterion,device)
