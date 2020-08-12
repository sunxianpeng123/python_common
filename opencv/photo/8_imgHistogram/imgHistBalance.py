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
# 直方图均衡化
#       图像的直方图是对图像对比度效果上的一种处理，旨在使得图像整体效果均匀，黑与白之间的各个像素级之间的点更均匀一点。

#       直方图均衡化：如果一副图像的像素占有很多的灰度级而且分布均匀，那么这样的图像往往有高对比度和多变的灰度色调。
#       直方图均衡化就是一种能仅靠输入图像直方图信息自动达到这种效果的变换函数。它的基本思想是对图像中像素个数多的灰度级进行展宽，
#       而对图像中像素个数少的灰度进行压缩，从而扩展像元取值的动态范围，提高了对比度和灰度色调的变化，使图像更加清晰。

#       直方图均衡化的三种情况,分别是:
#           灰度图像直方图均衡化
#           彩色图像直方图均衡化
#           YUV 直方图均衡化
# ==============================================================================
def imgHistBalanceGray(gray):
    """
    灰度图像直方图均衡化
    :param gray:
    :return:
    """
    # 1 灰度图的 均衡化
    equal = cv2.equalizeHist(gray)
    # 绘制 直方图
    plt.hist(gray.ravel(), 256)
    plt.figure()
    plt.hist(equal.ravel(), 256)
    plt.show()

def imgHistBalanceColors(img):
    """
    彩色图像直方图均衡化
    :param gray:
    :return:
    """
    # 2 彩色图的均衡化
    b, g, r = cv2.split(img)
    equal_b = cv2.equalizeHist(b)
    equal_g = cv2.equalizeHist(g)
    equal_r = cv2.equalizeHist(r)
    merge = cv2.merge([equal_b, equal_g, equal_r])
    showImgs([merge])

def imgHistBalanceCompare(gray):
    """
    直方图均衡化对比
    :param gray:
    :return:
    """
    equal = cv2.equalizeHist(gray)

    # 原始灰度图
    plt.subplot(2, 2, 1), plt.imshow(gray, 'gray'), plt.axis('off')
    # 均衡化后的图像
    plt.subplot(2, 2, 2), plt.imshow(equal, 'gray'), plt.axis('off')
    # 原始图像的灰度直方图
    plt.subplot(2, 2, 3), plt.hist(gray.ravel(), 256)
    # 均衡化后的灰度直方图
    plt.subplot(2, 2, 4), plt.hist(equal.ravel(), 256)
    plt.show()

if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 127, 255, 0)
    print("img1 shape = {}".format(img.shape))

    """0 == imgHistBalanceGray,,,1 == imgHistBalanceColors,,,2 == imgGuassianPyrUp"""
    runType = 2
    if runType == 0:
        print("================灰度图像直方图均衡化=================")
        imgHistBalanceGray(img_gray)
    elif runType == 1:
        print("================彩色图像直方图均衡化=================")
        imgHistBalanceColors(img)
    elif runType == 2:
        print("================直方图均衡化对比=================")
        imgHistBalanceCompare(img_gray)
    else:
        raise Exception("The Value Of runType Should In （0,1,2）")

