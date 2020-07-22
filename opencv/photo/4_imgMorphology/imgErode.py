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
def imgErode(gray):
    """
    图像腐蚀
    :param img:
    :return:腐蚀操作会把前景物体的边缘腐蚀掉。原理是卷积核沿着图像滑动，如果卷积核对应区域的图像像素值都是1，
    则卷积核中心对应的像素值保持不变，反之则全变成0，所以在图像边缘区域，部分为0，部分为1的区域都会变成0，再往里面则会保持不变。

    效果，靠近前景的像素被腐蚀为0，前景物体变小，图像白色区域减少，对于去除白噪声很有用，可以断开两个连接在一起的物体。（图像当中的白噪声大概意思就是随机噪声）
        erode dst = cv2.erode(src, kernel, iterations)
            参数
                src : 源图像
                kernel : 卷积核 kernel = np.ones((5, 5), np.uint8)
                iterations : 迭代次数
            返回值
                dst ： 处理结果
    """
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(binary, kernal, iterations=1)
    showImg(erosion)


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgErode(img_gray)
