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
#形态变换是基于图像形状的一些简单操作。它通常在二值化图像上执行。
#它需要两个输入，一个是我们的原始图像，第二个是称为结构元素或内核，它决定了操作的本质。两个基本的形态学运算符是侵蚀和膨胀。
#然后它的变体形式如Opening，Closing，Gradient等也发挥作用。我们将在以下图片的帮助下逐一看到它们：
# ==============================================================================
def imgDilation(gray):
    """
    图像膨胀
    :param img:
    :return:膨胀可以理解为腐蚀的反操作，膨胀的原理是：同样的卷积核沿着图像滑动，只要卷积核对应的图像像素值有一个是1，则这块区域全部变成1.
        因此它增加了图像中的白色区域或前景对象的大小增加。通常，在去除噪音的情况下，腐蚀之后再膨胀。因为，腐蚀会消除白噪声，
        但它也会缩小我们的物体，所以我们需要再扩大它。由于噪音消失了，它们不会再回来，但我们的物体区域会增加。它也可用于连接对象的破碎部分。
        dilate dst = cv2.dilate(src, kernel, iterations)
            参数
                src : 源图像
                kernel : 卷积核
                iterations : 迭代次数
            返回值
                dst ： 处理结果
    """
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(binary, kernal, iterations=1)
    showImg(dilation)


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),"lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgDilation(img_gray)
