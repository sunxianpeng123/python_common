# encoding: utf-8

"""
@author: sunxianpeng
@file: transfer_learning.py
@time: 2019/11/14 14:00
"""
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np

def initialize_model(model_name,num_classes,feature_extract,is_pretrained=True):
    model = None
    input_size = None
    if model_name == 'resnet':
        # 关于对pre-trained模型的使用和理解, https://blog.csdn.net/gbyy42299/article/details/78977826
        model = models.resnet18(pretrained=is_pretrained)
        # 提取的参数梯度不更新
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
#       model.fc是resnet的最后全连接层
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        # in_features 是全连接层的输入特征维度
        fc_in_num = model.fc.in_features
        model.fc = nn.Linear(fc_in_num,num_classes)
        # resnet18网络输入图片维度是224，resnet34，50，101，152也是
        input_size = 224
        return model,input_size





if __name__ == '__main__':
    input_size = 224
    data_dir = "./hymenoptera_data"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    data_transforms = {
        "train": transforms.Compose([transforms.RandomResizedCrop(input_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val":transforms.Compose([transforms.Resize(input_size),
                                   transforms.CenterCrop(input_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    print("Initializing Datasets and Dataloaders...")
    #ImageFolder -->> https://blog.csdn.net/qq_18649781/article/details/89215261
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
    print(image_datasets)
    # train_loader = DataLoader(image_datasets['train'],batch_size=batch_size,shuffle=True,num_workers=4)
    # val_loader = DataLoader(image_datasets['val'],batch_size=batch_size,shuffle=True,num_workers=4)
    dataloader_dict = {x:DataLoader(image_datasets[x],batch_size=batch_size,shuffle=True,num_workers=4) for x in ['train','val'] }
    # model
    model_name = 'resnet'
    num_classes = 2
    epochs = 5
    feature_extract = True
    # pre trained model
    model,input_size = initialize_model(model_name,num_classes,feature_extract,is_pretrained=True)
    model = model.to(device)
    # 需要更新的参数
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        # 需要更新的参数存放在此
        params_to_update=[]
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = torch.optim.SGD(params_to_update,lr=0.001,momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
#     train
    begin_time = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs - 1))
        print("-" * 10)
        for phrase in ['train','val']:
            running_loss = 0.
            running_acc = 0.
            if phrase == 'train':
                model.train()
            else :
                model.eval()
            for inputs ,labels in dataloader_dict[phrase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.autograd.set_grad_enabled(phrase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs,labels)
                _,preds = torch.max(outputs,dim=1)
                if phrase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_acc += torch.sum(preds.view(-1)==labels.view(-1)).item()
            epoch_loss = running_loss / len(dataloader_dict[phrase].dataset)
            epoch_acc = running_acc / len(dataloader_dict[phrase].dataset)
            print("{} Loss: {} Acc: {}".format(phrase, epoch_loss, epoch_acc))
            if phrase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phrase == 'val':
                val_acc_history.append(epoch_acc)
    time_used = time.time()- begin_time
    print("Training compete in {}m {}s".format(time_used // 60, time_used % 60))
    print("Best val Acc: {}".format(best_acc))
    model.load_state_dict(best_model_wts)
#     plt
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, epochs + 1), val_acc_history, label="Pretrained")
    # plt.plot(range(1, num_epochs + 1), scratch_hist, label="Scratch")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, epochs + 1, 1.0))
    plt.legend()
    plt.show()



