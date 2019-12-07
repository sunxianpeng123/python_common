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

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,1)
        self.conv2 = nn.Conv2d(20,50,5,1)
        self.full_c1 = nn.Linear(4*4*50,500)
        self.full_c2 = nn.Linear(500,10)

    def forward(self, x):
        conv1 = self.conv1(x)
        r1 = F.relu(conv1)
        pool1 = F.max_pool2d(r1,2,2)#12
        conv2 = self.conv2(pool1)
        r2 = F.relu(conv2)
        pool2 = F.max_pool2d(r2,2,2)#4
        v1 = pool2.view(-1,4*4*50)
        f1 = self.full_c1(v1)
        r3 = F.relu(f1)
        f2 = self.full_c2(r3)
        result = F.log_softmax(f2,dim=1)
        return result






if __name__ == '__main__':
    torch.manual_seed(53113)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    kwargs = {'num_workers':1,'pin_memory':True} if torch.cuda.is_available() else {}
    train_datesets =MNIST("./mnist",train=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])
                          ,download=True)
    test_datesets =MNIST("./mnist",train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ])
                          ,download=True)
    train_loader = DataLoader(dataset=train_datesets,batch_size=batch_size,shuffle=True,**kwargs)
    test_loader = DataLoader(dataset=test_datesets,batch_size=batch_size,shuffle=True,**kwargs)

    lr =0.01
    momentum = 0.9
    epochs = 5
    net =Net()
    model = net.to(device)
    opt =  torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum)
#     train
    for epoch in  range(epochs):
        model.train()
        for batch_idx,(data,target) in enumerate(train_loader):
            data,target = data.to(device),target.to(device)
            opt.zero_grad()
            out = model(data)
            loss = F.nll_loss(out,target)#avg loss
            loss.backward()
            opt.step()
            if batch_idx % 100 == 0:
                print("Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),  # 100*32
                    len(train_loader.dataset),  # 60000
                    100. * batch_idx / len(train_loader),  # len(train_loader)=60000/32=1875
                    loss.item()
                ))
#   test
        model.eval()
        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for data,target in test_loader:
                data,target = data.to(device),target.to(device)
                out = model(data)
                test_loss = F.nll_loss(out,target,reduction='sum')# sum loss
                pred = out.argmax(dim=1,keepdim=True)# get the index of the max log-probability
                # pred和target的维度不一样
                # pred.eq()相等返回1，不相等返回0，返回的tensor维度(32，1)。
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss = test_loss / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    save_model = True
    if save_model:
        torch.save(model.state_dict(),'simple_cnn.pt')






















