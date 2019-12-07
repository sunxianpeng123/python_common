# encoding: utf-8

"""
@author: sunxianpeng
@file: two_layer.py
@time: 2019/11/14 17:28
"""
import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self,batch_size,out1,num_classes):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear (batch_size,out1,bias=False)
        self.linear2 = nn.Linear(out1,num_classes,bias=False)

    def forward(self,x):
        l1 = self.linear1(x)
        r1 = torch.nn.functional.relu(l1)
        res = self.linear2(r1)
        return res

if __name__ == '__main__':
    N,D_in,H,D_out = 64,1000,100,10
    # 随机创建一些训练数据
    x = torch.randn(N,D_in)
    y = torch.randn(N,D_out)

    # 模型
    model = TwoLayerNet(D_in,H,D_out)
    # 方差损失
    loss_fn = nn.MSELoss(reduction="sum")
    learning_rate = 10**-4
    print("learning_rate = {}".format(learning_rate))
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    epochs = 500
    for epoch in range(epochs):
        pred = model(x)
        loss = loss_fn(pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("Epoch = {:.4f},Loss = {:.4f}".format(epoch,loss))
