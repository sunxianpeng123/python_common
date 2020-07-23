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
def imgCanny(gray):
    """
    Canny边缘检测
    :param img:
    :return:拉普拉斯算子可以使用二阶导数的形式定义，可假设其离散实现类似于二阶Sobel导数，事实上，OpenCV在计算拉普拉斯算子时直接调用Sobel 算子。
    Laplacian算子：图像中的边缘区域，像素值会发生“跳跃”，对这些像素求导，在其一阶导数在边缘位置为极值，这就是Sobel算子使用的原理——极值处就是边缘。
        cv2.Canny(image,            # 输入原图（必须为单通道图）
              threshold1,
              threshold2,       # 较大的阈值2用于检测图像中明显的边缘
              [, edges[,
              apertureSize[,    # apertureSize：Sobel算子的大小
              L2gradient ]]])   # 参数(布尔值)：
                                  true： 使用更精确的L2范数进行计算（即两个方向的倒数的平方和再开放），
                                  false：使用L1范数（直接将两个方向导数的绝对值相加）。
    """
    canny_1 = cv2.Canny(gray, 100, 200)
    canny_2 = cv2.Canny(gray, 64, 128)
    showImg(img=canny_1)
    showImg(img=canny_2)


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),"lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgCanny(img_gray)
