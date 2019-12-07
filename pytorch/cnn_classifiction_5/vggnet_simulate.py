# encoding: utf-8

"""
@author: sunxianpeng
@file: vggnet.py
@time: 2019/11/17 19:07
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

def vgg_block(num_convs,in_channels,out_channels):
    # 定义第一层
    net = [nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),nn.ReLU(True)]
    # 定义后面的多层
    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)

def vgg_stack(num_convs,channels):
    net = []
    # zip([1,2,3],[4,5,6]) ---> [(1, 4), (2, 5), (3, 6)]
    for n,c in zip(num_convs,channels):
        in_c = c[0]
        out_c = c[1]
        # num_convs = (1, 1, 2, 2, 2)
        # channels = ((3, 64), (64, 128), (128, 256), (256, 512), (512, 512))
        net.append(vgg_block(n,in_c,out_c))
    return nn.Sequential(*net)

class VggNet(nn.Module):
    def __init__(self,vgg_net):
        super(VggNet, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512*1*1,100),
            nn.ReLU(True),
            nn.Linear(100,10)
        )

    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # block_demo = vgg_block(3,64,128)
    # input_demo = torch.zeros(1,64,300,300)
    # output_deomo = block_demo(input_demo)
    # print(output_deomo.shape)

    train_set = CIFAR10('./cifar10', train=True, transform = data_tf)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    test_set = CIFAR10('./cifar10', train=False, transform = data_tf)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_convs = (1, 1, 2, 2, 2)
    channels = ((3, 64), (64, 128), (128, 256), (256, 512),(512, 512))
    vgg_net = vgg_stack(num_convs,channels)
    exit(0)

    # test_x = Variable(torch.zeros(1, 3, 256, 256))
    # test_y = vgg_net(test_x)
    # print(test_y.shape)

    net = VggNet(vgg_net)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss()
    train(net, train_data, test_data, 20, optimizer, criterion,device)