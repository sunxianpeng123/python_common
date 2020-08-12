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
# 形态变换是基于图像形状的一些简单操作。它通常在二值化图像上执行。
# 它需要两个输入，一个是我们的原始图像，第二个是称为结构元素或内核，它决定了操作的本质。两个基本的形态学运算符是侵蚀和膨胀。
# 然后它的变体形式如Opening，Closing，Gradient等也发挥作用。我们将在以下图片的帮助下逐一看到它们：
# ==============================================================================
def imgSobel(gray):
    """
    Sobel边缘检测
    :param img:
    :return:
    """
    # 1
    # 计算 深度-1，x方向梯度 1，即目标图像和原始图像都是256色域的图像，y方向梯度同理
    # 出现问题：会被截断，导致计算不出图像左侧的边界，
    soble_x1 = cv2.Sobel(gray, -1, 1, 0)
    showImg(soble_x1)
    # 2
    # 解决1中问题，y方向梯度同理
    soble_x2 = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    soble_x2 = cv2.convertScaleAbs(soble_x2)
    showImg(soble_x2)
    # 3
    # 同时计算x 和y 方向梯度（边界）,若同时计算 两个方向梯度边界，这种方式比较好
    soble_x3 = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    soble_y3 = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    soble_x3 = cv2.convertScaleAbs(soble_x3)
    soble_y3 = cv2.convertScaleAbs(soble_y3)
    soble_xy3 = cv2.addWeighted(soble_x3, 0.5, soble_y3, 0.5, 0)
    showImg(soble_xy3)
    # 4
    # 同时计算 x 和 y 方向梯度，但是这种方式计算出的结果不好
    # 出现问题：结果不好，最好使用 3 中的方式
    soble_xy4 = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
    soble_xy4 = cv2.convertScaleAbs(soble_xy4)
    showImg(soble_xy4)


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgSobel(img_gray)
