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
def imgHistMatplotlib(gray):
    """
    :param gray:
    :return:
    """
    array = gray.ravel()#ravel 将多维数组降为一维数组
    plt.hist(array,256)#256表示像素级，指[0,255]
    plt.show()



if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    imgHistMatplotlib(img_gray)
