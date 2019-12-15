# encoding: utf-8

"""
@author: sunxianpeng
@file: alexnet.py
@time: 2019/11/17 17:19
"""
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import CIFAR10

def data_tf(x):
    x = np.array(x,dtype='float32')/255
    x = (x - 0.5) / 0.5
    # 将 channel放在第一位,pytorch要求的格式
    x = x.transpose((2,0,1))
    x = torch.from_numpy(x)
    return x

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = pred_label.eq(label.view_as(pred_label)).sum().item()
    return num_correct / total

def train(net, train_data, valid_data, num_epochs, optimizer, criterion,device):
    prev_time = datetime.now()
    for epoch in range(num_epochs):
        print("epoch = {}".format(epoch))
        train_loss = 0
        train_acc = 0
        net = net.train()
        for im, label in train_data:
            im = im.to(device)  # (bs, 3, h, w)
            label =label.to(device)  # (bs, h, w)
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += get_acc(output, label)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                im = im.to(device)  # (bs, 3, h, w)
                label = label.to(device)  # (bs, h, w)
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                % (epoch, train_loss / len(train_data),
                   train_acc / len(train_data), valid_loss / len(valid_data),
                   valid_acc / len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
                         (epoch, train_loss / len(train_data),
                          train_acc / len(train_data)))
        prev_time = cur_time
        print(epoch_str + time_str)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层
        # (32 - 5 + 0)/ 1 + 1 = 28
        self.conv1 = nn.Sequential(
            nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=0)),
            nn.ReLU(True))
        # 第二层
        # (28 - 3 + 0) / 2 + 1 = 13,下取整
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
        # 第三层
        # (13 - 5 + 0)/1 + 1 = 9
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,5,1,0),
            nn.ReLU(True))
        # 第四层
        # (9 - 3 + 0)/ 2 + 1 = 4
        self.max_pool2 = nn.MaxPool2d(3,2,0)
        # 第五层
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4,384),
            nn.ReLU(True))
        # 第六层
        self.fc2 = nn.Sequential(
            nn.Linear(384,192),
            nn.ReLU(True))
        # 第七层
        self.fc3 = nn.Linear(192,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        #将矩阵拉平
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    alexnet = AlexNet()
    input_demo = torch.zeros(1,3,32,32)
    output_demo = alexnet(input_demo)
    print(output_demo.shape)

    train_set = CIFAR10('./cifar10',train=True,transform=data_tf,download=True)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./cifar10', train=False, transform=data_tf)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alexnet = alexnet.to(device)
    opt = torch.optim.SGD(alexnet.parameters(),lr = 0.001)
    loss_fn = nn.CrossEntropyLoss()
    train(alexnet,train_data,test_data,10,opt,loss_fn,device)


