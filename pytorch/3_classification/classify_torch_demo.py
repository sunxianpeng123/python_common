# encoding: utf-8

"""
@author: sunxianpeng
@file: classify_torch_demo.py
@time: 2019/11/4 19:56
"""

import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
# PyTorch无法直接处理图像，需要将图像转换成tensor。
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
# 虽然很容易实现softmax函数，我们将使用PyTorch中提供的实现，因为它适用于多维tensor（在我们的例子中是输出行列表）。
from torch.nn import functional as F


def show_one_dataset_image(datasets,i,show_image=False):
    """查看数据集中的某个图像"""
    image,label = datasets[i]
    if show_image is not False:
        plt.imshow(image[0])
        plt.show()
    return image,label

def plt_tensor_img(datasets,i):
    """使用tensor画出图像"""
    img_tensor,label = datasets[i]
    print(img_tensor.shape,label)
    # 颜色映射（cmap ='gray'），表示我们想要查看灰度图像。
    plt.imshow(img_tensor[0],cmap='gray')
    plt.show()

def split_indices(n,val_pct):
    """ 划分验证集和训练集
    split_indices随机地混洗数组索引0,1，... n-1，并从中为验证集分离出所需的部分。"""
    n_val = int(val_pct*n)
    # 混洗索引
    idxs = np.random.permutation(n)
    print(idxs)
    return idxs[n_val:],idxs[:n_val]

def loss_batch(model,loss_func,xb,yb,opt=None,metric=None):
    """
    增加我们之前定义的拟合函数，以使用每个epoch末尾的验证集来评估模型的准确性和损失。
    计算一批数据的损失如果提供了优化程序，则可以选择执行梯度下降更新步骤
    可选地使用预测和实际目标来计算度量（例如，准确度）
    :param model:
    :param loss_func:
    :param xb:
    :param yb:
    :param opt:
    :param mertic:
    :return:
    """
    preds = model(xb)
    loss=loss_func(preds,yb)#计算误差

    if opt is not None:
        loss.backward()#计算梯度
        opt.step()#更新参数
        opt.zero_grad()#重置梯度为0
    metric_result = None
    if metric is not None:
        metric_result = metric(preds,yb)
    return loss.item(),len(xb),metric_result

def evaluate(model,loss_fn,valid_dl,metric=None):
    """
    函数evaluate，它计算验证集的总体损失。
    :param model:
    :param loss_fn:
    :param valid_dl:
    :param metric:
    :return:
    """
    with torch.no_grad():
        # 对每个批次的数据进行训练，得到误差、数据量等数据
        result = [loss_batch(model,loss_fn,xb,yb,metric=metric) for xb,yb in valid_dl]
        # 将三者分开
        losses,nums,metrics = zip(*result)
        # 数据集总大小
        total = np.sum(nums)
        # 所有批次的平均误差
        total_loss = np.sum(np.multiply(losses,nums))
        avg_loss = total_loss / total

        avg_metric = None
        if metric is not None:
            tot_metric = np.sum(np.multiply(metrics,nums))
            avg_metric = tot_metric / total
        return avg_loss,total,avg_metric

def accuracy(outputs,labels):
    """定义精确度以直接操作整批输出，以便我们可以将其用作拟合度量"""
    _,preds = torch.max(outputs,dim=1)
    return torch.sum(preds == labels).item() / len(preds)

def fit(epochs,models,loss_fn,opt,train_dl,valid_dl,metric=None):
    """
    使用loss_batch和evaluate轻松定义拟合函数
    :param epoch:
    :param models:
    :param loss_fn:
    :param opt:
    :param train_dl:
    :param valid_dl:
    :param metric:
    :return:
    """
    for epoch in range(epochs):
        for xb,yb in train_dl:
            loss,_,_ = loss_batch(model,loss_fn,xb,yb,opt)
    result = evaluate(model,loss_fn,valid_dl,metric)
    val_loss ,total,val_metric = result
    if metric is None:
        print("Epoch[{}/{}],Loss:{:.4f}".format(epoch+1,epochs,val_loss))
    else:
        print("Epoch[{}/{}],Loss:{:.4f},{}:{:.4f}".format(epoch + 1, epochs, val_loss, metric.__name__, val_metric))

def predict_image(img,model):
    # img.unsqueeze只是在1x28x28张量的开始处添加另一个维度，
    # 使其成为1x1x28x28张量，模型将其视为包含单个图像的批处理。
    xb = img.unsqueeze(0)
    yb = model(xb)
    _,preds = torch.max(yb,dim=1)
    return preds[0].item()


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size,num_classes)
    def forward(self,xb):
        # 将图像打平，维度为（1,784
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

if __name__ == '__main__':
    # datasets = MNIST(root="data/",download=True)
    # PyTorch数据集允许我们指定一个或多个转换函数，这些函数在加载时应用于图像。
    # torchvision.transforms包含许多这样的预定义函数，我们将使用ToTensor变换将图像转换为PyTorchtensor。
    datasets = MNIST(root="F:\PythonProjects\python_study\deeplearning\pytouch",train=True,transform=transforms.ToTensor(), download=True)
    train_indices,val_indices = split_indices(len(datasets),0.2)
    print(len(train_indices),len(val_indices))

    batch_size = 100
    # 使用SubsetRandomSampler为每个创建PyTorch数据加载器
    # SubsetRandomSampler从给定的索引列表中随机采样元素，同时创建batch数据。
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(datasets,batch_size,sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(datasets,batch_size,sampler=val_sampler)
    # 每个1x28x28图像tensor需要在传递到模型之前被展平为大小为784（28 * 28）的tensor
    # 每个图像的输出是大小为10的tensor，tensor的每个元素表示特定目标标记（即0到9）的概率。
    # 图像的预测标签只是具有最高概率的标签
    input_size = 28 * 28
    num_classes = 10
    # 逻辑回归模型
    # model =nn.Linear(input_size,num_classes)
    # print(model.weight.shape)
    # print(model.bias.shape)

    # 请注意，模型不再具有.weight和.bias属性（因为它们现在位于.linear属性中），但它确实有一个.parameters方法，该方法返回包含权重和偏差的列表，并且可以使用PyTorch优化器。
    model = MnistModel()
    print("model.parameters() = ",model.parameters())
    # 分类问题常用的损失函数是交叉熵
    loss_fn = F.cross_entropy
    # 优化
    learning_rate = 0.001
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    fit(5,model,loss_fn,opt,train_loader,val_loader,accuracy)

    name = 'mnist-logistic.pth'
    torch.save(model.state_dict(),name)
    print(model.state_dict())
    model_saved = MnistModel()
    model_saved.load_state_dict(torch.load(name))
    print(model_saved.state_dict())
    """单个样本测试"""
    test_datasets = MNIST(root='data/',train=False,transform=transforms.ToTensor())
    image,label = show_one_dataset_image(test_datasets,0,False)
    print('Lable:',label,',Predicted:',predict_image(image,model))

    test_loader = DataLoader(test_datasets,batch_size=200)
    test_loss,total,test_acc= evaluate(model_saved,loss_fn,test_loader,metric=accuracy)
    print('model validate Loss:{:.4f}, Accuracy:{:.4f}'.format(test_loss,test_acc))







