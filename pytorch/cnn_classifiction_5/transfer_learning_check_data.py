# encoding: utf-8

"""
@author: sunxianpeng
@file: transfer_learning.py
@time: 2019/11/8 15:29
"""
import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import time
import os
import copy

print("Torchvision Version: ", torchvision.__version__)
"""
1、
    很多时候当我们需要训练一个新的图像分类任务，我们不会完全从一个随机的模型开始训练，而是利用_预训练_的模型来加速训练的过程。我们经常使用在ImageNet上的预训练模型。
    这是一种transfer learning的方法。我们常用以下两种方法做迁移学习。
        fine tuning: 从一个预训练模型开始，我们改变一些模型的架构，然后继续训练整个模型的参数。
        feature extraction: 我们不再改变与训练模型的参数，而是只更新我们改变过的部分模型参数。我们之所以叫它feature extraction是因为我们把预训练的CNN模型当做一个特征提取模型，利用提取出来的特征做来完成我们的训练任务。
2、
    以下是构建和训练迁移学习模型的基本步骤：
    （1）初始化预训练模型
    （2）把最后一层的输出层改变成我们想要分的类别总数
    （3）定义一个optimizer来更新参数
    （4）模型训练
"""

def check_one_image_in_loader(t_loader, title='Image'):
    batch_imgs = next(iter(t_loader))[0]
    print('batch_imgs.shape = {}'.format(batch_imgs.shape))
    img_tensor = batch_imgs[8]
    img = img_tensor.cpu().clone()  # we clone the tensor to not do changes on it
    # img = img.squeeze(0)  # remove the fake batch dimension,这个.squeeze(0)看不懂，去掉也可以运行
    print('img.shape = {}'.format(img.shape))
    # transforms：torchvision的子模块，常用的图像操作
    # .ToPILImage() 把tensor或数组转换成图像
    # 详细转换过程可以看这个：https://blog.csdn.net/qq_37385726/article/details/81811466
    # 详细了解看这个：https://blog.csdn.net/SZuoDao/article/details/52973621
    image =transforms.ToPILImage()(img)# unloader(img)  # tensor转换成图像
    plt.imshow(image)
    plt.title(title)
    # 可以去掉看看，只是延迟显示作用
    plt.pause(1)  # pause a bit so that plots are updated
    plt.show()

if __name__ == '__main__':
    input_size = 224
    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    data_dir = "F:\PythonProjects\python_common\pytorch\data\hymenoptera_data"
    # Batch size for training (change depending on how much memory you have)
    batch_size = 32
    # 蜜蜂和蚂蚁数据集不会自动下载，请到群文件下载，并放在当前代码目录下
    all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transforms.Compose([
                                        transforms.RandomResizedCrop(input_size),  # 把每张图片变成resnet需要输入的维度224
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                    ]))
    # 训练数据分batch，变成tensor迭代器
    loader = torch.utils.data.DataLoader(all_imgs, batch_size=batch_size, shuffle=True, num_workers=4)
    # 这个img是一个batch的tensor，torch.Size([32, 3, 224, 224])，三十二张224*224*3的图片
    check_one_image_in_loader(loader)


