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
    #reshape成 batch_size,input_channel,图像高度，图像宽度,此处为一张单通道图片
    tensor = torch.from_numpy(arr.reshape((1,1,arr.shape[0],arr.shape[1])))
    print(tensor.shape)
    """=========== 1 ============="""
    pool1 = nn.MaxPool2d(2,2)
    print('before	max	pool,	image	shape:	{}	x	{}'
          .format(tensor.shape[2], tensor.shape[3]))
    small_im1 = pool1(Variable(tensor))
    small_im1 = small_im1.data.squeeze().numpy()
    print('after	max	pool,	image	shape:	{}	x	{}	'
          .format(small_im1.shape[0],small_im1.shape[1]))
    show_img(arr)
    show_img(small_im1)
    """=========== 2 ============="""
    print('before	max	pool,	image	shape:	{}	x	{}'
          .format(tensor.shape[2], tensor.shape[3]))
    small_im2 = F.max_pool2d(Variable(tensor), 2, 2)
    small_im2 = small_im2.data.squeeze().numpy()
    print('after	max	pool,	image	shape:	{}	x	{}	'
          .format(small_im1.shape[0],small_im1.shape[1]))
    show_img(small_im2)