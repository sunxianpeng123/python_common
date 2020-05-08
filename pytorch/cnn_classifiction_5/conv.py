# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_helloworld_1.py
@time: 2019/11/17 16:19
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import  Image
import  matplotlib.pyplot as plt

def show_img(arr):
    plt.imshow(arr.astype('uint8'),cmap='gray')
    plt.show()

if __name__ == '__main__':
    # PIL读取图片并处理
    img = Image.open('lena.jpg')
    gray = img.convert('L')
    arr = np.array(gray,dtype='float32')
    show_img(arr)
    #reshape成 batch_size,input_channel,图像高度，图像宽度,此处为一张单通道图片
    tensor = torch.from_numpy(arr.reshape((1,1,arr.shape[0],arr.shape[1])))
    print(tensor.shape)
    """=========== 1 ============="""
    #定义卷积
    conv1 = nn.Conv2d(1,1,1,bias=True)
    # 定义轮廓检测算子
    sobel_kernel = np.array([[-1,	-1,	-1],
                             [-1,	8,	-1],
                             [-1,	-1,	-1]],	dtype='float32')
    # sobel_kernel.dtype=np.float32
    print(sobel_kernel.shape)
    sobel_kernel = sobel_kernel.reshape((1,1,3,3))
    # 给卷积的kernel赋值
    conv1.weight.data = torch.from_numpy(sobel_kernel)
    # 作用在图片上
    edge1 = conv1(tensor)
    # 将输出转换为图片的格式
    edge1 = edge1.data.squeeze().numpy()
    show_img(edge1)
    """=========== 2 ============="""
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge2 = F.conv2d((Variable(tensor),weight))
    edge2 = edge2.data.squeeze().numpy()
    show_img(edge2)


