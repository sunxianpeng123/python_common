# encoding: utf-8

"""
@author: sunxianpeng
@file: transfer_learning.py
@time: 2019/11/8 15:29
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import time
import os
import copy
# print("Torchvision Version: ",torchvision.__version__)

def check_train_data(dataloaders_dict):
    """查看一个批次的数据"""
    inputs, labels = next(iter(dataloaders_dict["train"]))  # 一个batch
    print("inputs.shape = {}".format(inputs.shape))#torch.Size([32, 3, 224, 224])
    # tensor([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    print("labels = {}".format(labels))
    """查看所有训练数据批次的batch_size大小"""
    for inputs, labels in dataloaders_dict["train"]:
        # print(inputs)
        # print(labels)
        print(labels.size())  # 最后一个batch不足32

def check_model_ft_info(model_ft):
    print("model_ft = {}".format(model_ft))
    print("next(iter(model_ft.named_parameters())) = {}".format(next(iter(model_ft.named_parameters()))))
     # 是元组，只有两个值
    print("len(next(iter(model_ft.named_parameters()))) = {}".format(len(next(iter(model_ft.named_parameters())))))
    for name, param in model_ft.named_parameters():
        print(name)  # 看下都有哪些参数


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False #提取的参数梯度不更新


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = None
    if model_name == "resnet":
        # 如果True，从imagenet上返回预训练的模型和参数
        #关于对pre-trained模型的使用和理解, https://blog.csdn.net/gbyy42299/article/details/78977826
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)  # 提取的参数梯度不更新
        # print(model_ft) 可以打印看下
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc是resnet的最后全连接层
        # (fc): Linear(in_features=512, out_features=1000, bias=True)
        # in_features 是全连接层的输入特征维度
        # print(num_ftrs)
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # out_features=1000 改为 num_classes=2
        input_size = 224  # resnet18网络输入图片维度是224，resnet34，50，101，152也是
    return model_ft, input_size

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5):
    # global device
    """ #训练测试合一起"""
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())  # 深拷贝上面resnet模型参数
    # .copy和.deepcopy区别看这个：https://blog.csdn.net/u011630575/article/details/78604226
    best_acc = 0.
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # torch.autograd.set_grad_enabled梯度管理器，可设置为打开或关闭
                # phase=="train"是True和False，双等号要注意
                with torch.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                # 返回每一行最大的数和索引，prds的位置是索引的位置
                # 也可以preds = outputs.argmax(dim=1)
                _, preds = torch.max(outputs, 1)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # 交叉熵损失函数是平均过的,所以要与本批数据个数相乘
                running_loss += loss.item() * inputs.size(0)
                # .view(-1)展开到一维，并自己计算
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 模型变好，就拷贝更新后的模型参数
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                # 记录每个epoch验证集的准确率
                val_acc_history.append(epoch_acc)
        print()
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {}".format(best_acc))
    # 把最新的参数复制到model中
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def train_all_parameter():
    """训练所有参数，包括resnet的参数"""
    # Initialize the non-pretrained version of the model used for this run
    scratch_model, _ = initialize_model(model_name,
                                        num_classes,
                                        feature_extract=False,  # 所有参数都训练
                                        use_pretrained=False)  # 不要imagenet的参数
    scratch_model = scratch_model.to(device)
    scratch_optimizer = optim.SGD(scratch_model.parameters(),lr=0.001, momentum=0.9)
    scratch_criterion = nn.CrossEntropyLoss()
    _, scratch_hist = train_model(scratch_model,dataloaders_dict,scratch_criterion, scratch_optimizer,num_epochs=num_epochs)


if __name__ == '__main__':
    input_size = 224#输入图片的大小
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "./hymenoptera_data"
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    # 把迭代器存放到字典里作为value，key是train和val，后面调用key即可。
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
    # 输出结构
    print(dataloaders_dict)
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 查看数据
    # check_train_data(dataloaders_dict)
    """加载resnet模型并修改全连接层"""
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"
    # Number of classes in the dataset
    num_classes = 2
    # Number of epochs to train for
    num_epochs = 2
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True  # 只更新修改的层
    # 模型
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # 查看resnet模型相关信息
    # check_model_ft_info(model_ft)
    # Send the model to GPU
    model_ft = model_ft.to(device)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()  # 需要更新的参数
    print("Params to learn:")
    if feature_extract:
        params_to_update = []  # 需要更新的参数存放在此
        for name, param in model_ft.named_parameters():
            # model_ft.named_parameters()有啥看上面cell
            if param.requires_grad == True:
                # 这里要知道全连接层之前的层param.requires_grad == Flase
                # 后面加的全连接层param.requires_grad == True
                params_to_update.append(param)
                print("\t", name)
    else:  # 否则，所有的参数都会更新
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)  # 定义优化器
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()  # 定义损失函数
    # Train and evaluate
    model_ft, ohist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
    # print("model_ft = {}".format(model_ft))
    print("ohist = {}".format(ohist))

    """训练所有参数"""
    # train_all_parameter()
    # Plot the training curves of validation accuracy vs. number
    #  of training epochs for the transfer learning method and
    #  the model trained from scratch
    # ohist = []
    # shist = []

    # ohist = [h.cpu().numpy() for h in ohist]
    # shist = [h.cpu().numpy() for h in scratch_hist]

    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    # plt.plot(range(1, num_epochs + 1), scratch_hist, label="Scratch")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()









