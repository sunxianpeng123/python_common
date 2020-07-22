# encoding: utf-8

"""
@author: sunxianpeng
@file: imgOpen.py
@time: 2020/7/22 16:18
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
# morphologyEx result = cv2.morphologyEx(img, type, kernel)
#           参数
#                 img : 源图像
#                 type : 图像形态学方法
#                     1、开运算 cv2.MORPH_OPEN :
#                         开运算（image） = 膨胀（腐蚀（image））
#                             图像被腐蚀后，去除了噪声，但是会压缩图像。
#                             对腐蚀过的图像，进行膨胀操作，可以去除噪声，并保持原有形状
#                     2、闭运算 cv2.MORPH_CLOSE
#                         闭运算（image） = 腐蚀(膨胀（image）)
#                             先膨胀，后腐蚀
#                             它有助于关闭前景物体内的小孔，或物体上的小黑点
#                     3、梯度运算 cv2.MORPH_GRADIENT
#                         梯度（image） = 膨胀（image） - 腐蚀（image）
#                             膨胀图像，腐蚀图像，得到轮廓图像
#                     4、礼帽运算 cv2.MORPH_TOPHAT
#                         礼帽（image） = image - 开运算(image)
#                             得到噪声图像
#                     5、黑帽运算 cv2.MORPH_BLACKHAT
#                         黑帽（image） = 闭运算（image） - image
#                             得到图像内部的小孔，或者景色中的小黑点
#             kernel ： 卷积核
#         返回值
#             result ： morphologyEx函数运算结果
# ==============================================================================
def imgGradient(gray):
    """
    梯度运算
    :param img:
    :return:就是图像的膨胀和腐蚀之后两张图形的差，其结果就像原图形的轮廓
        其效果等同于膨胀减去腐蚀。
    """
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernal = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernal)
    showImg(gradient)

if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgGradient(img_gray)
