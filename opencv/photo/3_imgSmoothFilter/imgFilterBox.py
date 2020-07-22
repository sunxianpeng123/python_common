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
def imgBoxFilter(gray):
    """
    方框滤波器
    :param img:
    :return:OpenCV中有一个专门的平均滤波模板供使用------归一化卷积模板，所有的滤波模板都是使卷积框覆盖区域所有像素点与模板相乘后得到的值作为中心像素的值。
            OpenCV中均值模板可以用 cv2.blur 和 cv2.boxFilter ,比如一个3*3的模板其实就可以如下表示；

            模板大小m*n是可以设置的。如果不想要前面的1/9，可以使用非归一化模板cv2.boxFitter。
            cv2.blur(src,ksize,dst,anchor,borderType)
                src: 输入图像对象矩阵,可以为单通道或多通道
                ksize:高斯卷积核的大小，格式为(宽，高)
                dst:输出图像矩阵,大小和数据类型都与src相同
                anchor：卷积核锚点，默认(-1,-1)表示卷积核的中心位置
                borderType:填充边界类型
    """
    box = cv2.boxFilter(gray, -1, (5, 5), normalize=True)

    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(gray, 'gray')  # 默认彩色，指定为gray
    plt.subplot(1, 2, 2), plt.imshow(box, 'gray')

    plt.show()



if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgBoxFilter(img_gray)
