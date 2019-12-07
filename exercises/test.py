# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2019/11/14 14:12
"""
import torch
import torchvision
from torchvision import datasets,transforms

import skimage.io as io
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def read_image():
    img_skimage =io.imread('test.jpg')
    img_cv = cv2.imread('test.jpg')
    print("cv2 or skimage img shape ={}".format(img_cv.shape))
    img_pil = Image.open('test.jpg')
    img_pil_1 = np.array(img_pil)  # (H x W x C), [0, 255], RGB
    print("img_pil shape ={}".format(img_pil_1.shape))
#    trans to tensor
    # numpy image: H x W x C
    # torch image: C x H x W
    # np.transpose( xxx,  (2, 0, 1))   # 将 H x W x C 转化为 C x H x W

    return  img_pil




if __name__ == '__main__':
    trans = transforms.Compose([
        transforms.CenterCrop(10),
        transforms.ToTensor(),
    ])
    img = read_image()

    resize = transforms.Resize()

    # 定义中心切割
    centerCrop = transforms.CenterCrop((img.size[0] / 2, img.size[1] / 2))
    imgccrop = centerCrop(img)
    print(type(imgccrop))
    plt.imshow(imgccrop)
    plt.show()