# encoding: utf-8

"""
@author: sunxianpeng
@file: imgBinary.py
@time: 2020/7/22 11:59
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def showImg(img):
    """显示单个图像"""
    plt.imshow(img)
    plt.show()


def showImgs(imgs):
    """传入图片的List，显示多个图片，"""
    plt.figure()
    img_index = 1
    for img in imgs:
        # print(img.shape)
        plt.subplot(1, len(imgs), img_index)
        plt.imshow(img)
        img_index = img_index + 1
    plt.show()


# ==============================================================================
# 读取图片
# ==============================================================================
def readImg(image_path, path_has_chinese=True):
    """
    读取图片
    :param image_path:图片路径
    :param path_has_chinese: 路径中是否有中文，默认有中文
    :return:
    """
    if not path_has_chinese:
        img = cv2.imread(image_path)
    else:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    # print(isinstance(img, type(None)))
    # 判断img是否为None
    if isinstance(img, type(None)) == True:  raise Exception("File Not Found!!")
    return img


# ==============================================================================
# 一、Matplotlib 绘制直方图
# 1、概念
#       直方图简单来说就是图像中每个像素值的个数统计，比如说一副灰度图中像素值为0的有多少个，1的多少个……直方图是一种分析图片的手段：
#       归一化直方图：统计不同像素值的个数在图像中出现的概率。
#       要理解直方图，绕不开“亮度”这个概念。人们把照片的亮度分为0到255共256个数值，数值越大，代表的亮度越高。
#       其中0代表纯黑色的最暗区域，255表示最亮的纯白色，而中间的数字就是不同亮度的灰色。人们还进一步把这些亮度分为了5个区域，分别是黑色，阴影，中间调，高光和白色。
# 2、几个重要参数
#       dims：要计算的通道数，对于灰度图 dims=1
#       range：要计算的像素值范围，一般为[0,256]（不包括256）
#       bins：子区段数目，如果我们统计0255每个像素值，bins=256；如果划分区间，比如015, 1631…240255这样16个区间，bins=16
# ==============================================================================
def imgHistWithMask(img, mask):
    """
    1、图像掩模主要用于：
        ①提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。
        ②屏蔽作用,用掩模对图像上某些区域作屏蔽,使其不参加处理或不参加处理参数的计算,或仅对屏蔽区作处理或统计。
        ③结构特征提取,用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。
        ④特殊形状图像的制作。
    2、掩膜就是一个区域大小，表示你接下来的直方图统计就是这个区域的像素统计
    :param gray:
    :return:
    """

    # 返回图像中每个像素值的个数
    histB = cv2.calcHist([img], [0], mask=None, histSize=[256], ranges=[0, 255])
    histB_m = cv2.calcHist([img], [0], mask=mask, histSize=[256], ranges=[0, 255])

    print('hist type ={}\nhist size = {}\nhist shape = {}'
          .format(type(histB), histB.size, histB.shape))

    plt.plot(histB, color='b')
    plt.plot(histB_m, color='g')
    plt.show()


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 127, 255, 0)
    print("img1 shape = {}".format(img.shape))
    # 掩膜
    mask = np.zeros(img_gray.shape, np.uint8)
    mask[200:400, 200:400] = 255
    imgHistWithMask(img, mask)
