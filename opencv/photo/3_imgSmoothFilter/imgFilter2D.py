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
# 运用它，首先就要了解它，什么是平滑滤波？
#       平滑滤波是低频增强的空间域滤波技术。它的目的有两类：
#       一类是模糊；另一类是消除噪音。
#       空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。
# 在看一下滤波的目的：
# 滤波的本义是指信号有各种频率的成分,滤掉不想要的成分,即为滤掉常说的噪声,留下想要的成分.这即是滤波的过程,也是目的。
#     抽出对象的特征作为图像识别的特征模式；
#     为适应图像处理的要求，消除图像数字化时所混入的噪声。
# ==============================================================================
def imgFilter2D(gray, k_max=5):
    """
    2D滤波器cv2.filter2D( )
    :param img:
    :return:Opencv提供的一个通用的2D滤波函数为cv2.filter2D()，滤波函数的使用需要一个核模板，对图像的滤波操作过程为：将和模板放在图像的一个像素A上，
            求与之对应的图像上的每个像素点的和，核不同，得到的结果不同，而滤波的使用核心也是对于这个核模板的使用，需要注意的是，该滤波函数是单通道运算的，
            也就是说对于彩色图像的滤波，需要将彩色图像的各个通道提取出来，对各个通道分别滤波才行。
        对于2D图像可以进行低通或者高通滤波操作
        　　　　低通滤波（LPF）：有利于去噪，模糊图像
        　　　　高通滤波（HPF）：有利于找到图像边界
        使用自定义内核对图像进行卷积。该功能将任意线性滤波器应用于图像。支持就地操作。当光圈部分位于图像外部时，该功能会根据指定的边框模式插入异常像素值。
        cv2.filter2D(src,dst,ddepth,kernel,anchor=(-1,-1),delta=0,borderType=cv2.BORDER_DEFAULT)
            src: 输入图像对象矩阵
            dst:输出图像矩阵
            ddepth:输出矩阵的数值类型
            kernel:卷积核
            anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
            delat:卷积完后相加的常数
            borderType:填充边界类型
        参数：
            参数 	描述
            src 	原图像
            dst 	目标图像，与原图像尺寸和通过数相同
            ddepth 	输出图像深度（通道数），-1表示和原图像一致
            kernel 	卷积核（或相当于相关核），单通道浮点矩阵;如果要将不同的内核应用于不同的通道，请使用拆分将图像拆分为单独的颜色平面，然后单独处理它们。
            anchor 	内核的锚点，指示内核中过滤点的相对位置;锚应位于内核中;默认值（-1，-1）表示锚位于内核中心。
            detal 	在将它们存储在dst中之前，将可选值添加到已过滤的像素中。类似于偏置。
            borderType 	像素外推法，参见BorderTypes

    """
    gray1 = np.float32(gray)  # 转化数值类型

    # kernel = np.ones((5, 5), np.float32) / 25
    kernel = np.array([[0, -1, 0],
                           [-1, k_max, -1],
                           [0, -1, 0]])
    dst = cv2.filter2D(gray, -1, kernel)
    # cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
    # 输出图像与输入图像大小相同
    # 中间的数为-1，输出数值格式的相同plt.figure()

    plt.subplot(1, 2, 1), plt.imshow(gray1, 'gray')  # 默认彩色，另一种彩色bgr
    plt.subplot(1, 2, 2), plt.imshow(dst, 'gray')
    plt.show()


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgFilter2D(img_gray)
