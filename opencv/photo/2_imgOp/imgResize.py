# encoding: utf-8

"""
@author: sunxianpeng
@file: imgResizeRotate.py
@time: 2020/7/21 14:30
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
# 读取操作
# ==============================================================================
def imgResize(img1, img2):
    """
    图像缩放
    :param img1:
    :param img2:
    :param
    OpenCV转换成PIL.Image格式
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    Image转换成OpenCV格式
        img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
    :return:
    resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )
    参数说明：
        （1）src - 原图
        （2）dst - 目标图像。当参数dsize不为0时，dst的大小为size；否则，它的大小需要根据src的大小，参数fx和fy决定。dst的类型（type）和src图像相同
        （3）dsize - 目标图像大小。指定方式（列大小，行大小），注意顺序
            当dsize为0时，它可以通过以下公式计算得出：
            所以，参数dsize和参数(fx, fy)不能够同时为0
        （4）fx - 水平轴上的比例因子(即可以计算出列数)。当它为0时，计算公式如下：
        （5）fy - 垂直轴上的比例因子（即可以计算出行数）。当它为0时，计算公式如下：
        （6）interpolation - 插值方法。共有5种：
            １）INTER_NEAREST - 最近邻插值法
            ２）INTER_LINEAR - 双线性插值法（默认）
            ３）INTER_AREA - 基于局部像素的重采样（resampling using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
            ４）INTER_CUBIC - 基于4x4像素邻域的3次插值法
            ５）INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值

    """
    return None

if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img1 = readImg(read_path_lena,False)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img1.shape))
    print("=================================")
    # 1、缩放成指定大小
    img2 = cv2.resize(img1, (200, 100))
    print("img2 shape = {}".format(img2.shape))
    showImg(img2)

    # 2、根据比例，指定缩放后图片的大小，指定缩放后图片长、宽的比例
    rows, cols, chn = img1.shape
    size3 = (round(cols * 0.5), round(rows * 1.5))  #
    img3 = cv2.resize(img1, size3)
    print("img3 shape = {}".format(img3.shape))
    showImg(img3)
    # 3、指定比例缩放
    fx = 0.5  # 水平方向变成原来0.5倍
    fy = 1.3  # 竖直方向变成原来1.3倍
    img4 = cv2.resize(img1, dsize=None, fx=fx, fy=fy)
    print("img4 shape = {}".format(img4.shape))
    showImg(img4)
