# encoding: utf-8

"""
@author: sunxianpeng
@file: deepth_network_v1.py
@time: 2019/11/23 17:04
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def check_data():
    train_set = mnist.MNIST(mnist_path,train=True,download=True)
    test_set = mnist.MNIST(mnist_path,train=False,download=True)
    one_data,one_label = train_set[0]
    print(one_data)
    print(one_label)
    one_data = np.array(one_data,dtype='float32')
    print(one_data.shape)

def data_transform(x):
    x = np.array(x,dtype='float32') / 255
    x = (x - 0.5) / 0.5  # 标准化
    x = x.reshape((-1,))  # 拉平
    x = torch.from_numpy(x)
    return x

def getModel():
    # 四层
    net = nn.Sequential(
        nn.Linear(784, 400),
        nn.ReLU(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return net




if __name__ == '__main__':
    # check_data()
    epochs = 3
    learning_rate = 0.01

    mnist_path = r"F:\PythonProjects\python_study\pytouch\MNIST"
    train_set = mnist.MNIST(mnist_path,train=True,transform=data_transform,download=True)
    test_set = mnist.MNIST(mnist_path,train=False,transform=data_transform,download=True)
    train_data = DataLoader(train_set,batch_size=64,shuffle=True)
    test_data = DataLoader(test_set,batch_size=64,shuffle=True)

    modle = getModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modle.parameters(),lr=learning_rate)

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        modle.train()# 切换模型为训练模式
        for batch_imgs,label in train_data:
            # print(batch_imgs.shape)
            pred = modle(batch_imgs)
            loss = criterion(pred,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #
            train_loss += loss
            # 计算本批次分类准确率
            _,pred = pred.max(dim=1)#取出概率最大的分类
            num_correct = (pred == label).sum().item()
            acc = num_correct / batch_imgs.shape[0]
            train_acc += acc
        # 本轮训练的损失和准确率
        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))

        eval_loss = 0.
        eval_acc = 0.
        modle.eval()#切换模型为预测模式
        for batch_imgs,label in test_data:
            pred = modle(batch_imgs)
            loss = criterion(pred,label)
            eval_loss += loss
            _,pred = pred.max(dim=1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / batch_imgs.shape[0]
            eval_acc += acc
        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train	Loss: {:.6f}, Train	Acc: {:.6f}, Eval Loss:	{:.6f},	EvalAcc: {:.6f}'
            .format(epoch, train_loss / len(train_data), train_acc / len(train_data),
                    eval_loss / len(test_data), eval_acc / len(test_data)))

    # 画出训练数据和验证数据的 loss 曲线和 acc 曲线
    plt.plot(np.arange(len(losses)), losses,color='red',label='train loss')
    plt.plot(np.arange(len(acces)), acces, color='black', label='train acc')
    plt.plot(np.arange(len(eval_losses)), eval_losses, color='blue', label='test loss')
    plt.plot(np.arange(len(eval_acces)), eval_acces, color='yellow', label='test acc')
    plt.legend()
    plt.show()


