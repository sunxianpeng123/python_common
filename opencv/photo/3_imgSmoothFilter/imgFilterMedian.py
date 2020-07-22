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
def imgMedianFilter(gray):
    """
    中值滤波器
    :param img:
    :return:中值滤波模板就是用卷积框中像素的中值代替中心值，达到去噪声的目的。这个模板一般用于去除椒盐噪声。
            前面的滤波器都是用计算得到的一个新值来取代中心像素的值，而中值滤波是用中心像素周围（也可以使他本身）的值来取代他，卷积核的大小也是个奇数。
             cv2.medianBlur(src,ksize,dst)
                src: 输入图像对象矩阵,可以为单通道或多通道
                ksize:核的大小，格式为 3      #注意不是（3,3）
                dst:输出图像矩阵,大小和数据类型都与src相同
    """
    # 添加点噪声
    for i in range(2000):
        temp_x = np.random.randint(0, gray.shape[0])
        temp_y = np.random.randint(0, gray.shape[1])
        gray[temp_x][temp_y] = 255
    medina = cv2.medianBlur(gray, 5)
    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(gray, 'gray')  # 默认彩色，指定为gray
    plt.subplot(1, 2, 2), plt.imshow(medina, 'gray')

    plt.show()




if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgMedianFilter(img_gray)
