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
def imgBilateralFilter(gray):
    """
    中值滤波器
    :param img:
    :return:相比于上面几种平滑算法，双边滤波在平滑的同时还能保持图像中物体的轮廓信息。
        双边滤波在高斯平滑的基础上引入了灰度值相似性权重因子，所以在构建其卷积核核时，
        要同时考虑空间距离权重和灰度值相似性权重。在进行卷积时，每个位置的邻域内，根据和锚点的距离d构建距离权重模板，
        根据和锚点灰度值差异r构建灰度值权重模板，结合两个模板生成该位置的卷积核。
        opencv中的bilateralFilter()函数实现了双边滤波，其参数对应如下：
            dst = cv2.bilateralFilter(src,d,sigmaColor,sigmaSpace,borderType)
                src: 输入图像对象矩阵,可以为单通道或多通道
                d:用来计算卷积核的领域直径，如果d<=0，从sigmaSpace计算d
                sigmaColor：颜色空间滤波器标准偏差值，决定多少差值之内的像素会被计算（构建灰度值模板）
                sigmaSpace:坐标空间中滤波器标准偏差值。如果d>0，设置不起作用，否则根据它来计算d值（构建距离权重模板）
        双边滤波函数后面的两个参数设置的越大，图像的去噪越多，但随之而来的是图像ji将变得模糊，所以根据需要调整好后两个参数的大小。

    """
    # 添加点噪声
    # 添加点噪声
    # 添加点噪声
    for i in range(2000):
        temp_x = np.random.randint(0, gray.shape[0])
        temp_y = np.random.randint(0, gray.shape[1])
        gray[temp_x][temp_y] = 255
    # 9---滤波领域直径
    # 后面两个数字：空间高斯函数标准差，灰度值相似性标准差
    bilater = cv2.bilateralFilter(gray, 9, 75, 75)

    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(gray, 'gray')  # 默认彩色，指定为gray
    plt.subplot(1, 2, 2), plt.imshow(bilater, 'gray')

    plt.show()





if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgBilateralFilter(img_gray)
