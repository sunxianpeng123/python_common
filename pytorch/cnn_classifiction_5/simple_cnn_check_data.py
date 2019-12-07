# encoding: utf-8

"""
@author: sunxianpeng
@file: check_data.py
@time: 2019/11/8 14:33
"""
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# torchvision是独立于pytorch的关于图像操作的一些方便工具库。
# torchvision的详细介绍在：https://pypi.org/project/torchvision/0.1.8/
# torchvision主要包括一下几个包：
# vision.datasets : 几个常用视觉数据集，可以下载和加载
# vision.models : 流行的模型，例如 AlexNet, VGG, and ResNet 以及 与训练好的参数。
# vision.transforms : 常用的图像操作，例如：随机切割，旋转等。
# vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网。

# for d in mnist_data:
#     x = d[0].data
#     print(x)
#     plt.imshow(x[0])
#     plt.show()
#     exit(0)

def show_one_dataset_image(datasets,i,show_image=False):
    """查看数据集中的某个图像"""
    image,label = datasets[i]# 和下面形式相同
    # image = datasets[0][0].data
    # label = datasets[0][1]
    image = image[0]
    if show_image is not False:
        plt.imshow(image)
        plt.show()
    return image,label

if __name__ == '__main__':
    # torch.manual_seed(53113)  # cpu随机种子
    #
    # # 没gpu下面可以忽略
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    # batch_size = test_batch_size = 32
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #
    # # torch.utils.data.DataLoader在训练模型时使用到此函数，用来把训练数据分成多个batch，
    # # 此函数每次抛出一个batch数据，直至把所有的数据都抛出，也就是个数据迭代器。
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./mnist_data',
    #                    train=True,  # 如果true，从training.pt创建数据集
    #                    download=True,  # 如果ture，从网上自动下载
    #                    # transform 接受一个图像返回变换后的图像的函数，相当于图像先预处理下
    #                    # 常用的操作如 ToTensor, RandomCrop，Normalize等.
    #                    # 他们可以通过transforms.Compose被组合在一起
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        # .ToTensor()将shape为(H, W, C)的nump.ndarray或img转为shape为(C, H, W)的tensor，
    #                        # 其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可。
    #                        transforms.Normalize((0.1307,), (0.3081,))  # 所有图片像素均值和方差,https://blog.csdn.net/Harpoon_fly/article/details/84987589
    #                        # .Normalize作用就是.ToTensor将输入归一化到(0,1)后，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1)
    #                    ])),  # 第一个参数dataset：数据集
    #     batch_size=batch_size,
    #     shuffle=True,  # 随机打乱数据
    #     **kwargs)  ##kwargs是上面gpu的设置
    #
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./mnist_data',
    #                    train=False,  # 如果False，从test.pt创建数据集
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=test_batch_size,
    #     shuffle=True,
    #     **kwargs)
    #
    mnist_data = datasets.MNIST("./mnist_data", train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))

    print("mnist_data info = {}".format(mnist_data))
    image, label = show_one_dataset_image(mnist_data, 0, True)
    data = [d[0].data.cpu().numpy() for d in mnist_data]



