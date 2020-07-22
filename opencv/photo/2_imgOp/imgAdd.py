# encoding: utf-8

"""
@author: sunxianpeng
@file: read_write.py
@time: 2020/7/21 10:33
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
def imgAdd(img1, img2, use_numpy_or_opencv='opencv'):
    """
    图像加法
    :param img1:
    :param img2:
    :param use_numpy_or_opencv:使用numpy做图像加法还是使用opencv做图像加法
    :return:
    """
    img = None
    if use_numpy_or_opencv == 'numpy':
        # 若某个像素相加之和大于255，则相加后的值除以 255取余
        img = img1 + img2
    if use_numpy_or_opencv == 'opencv':
        # 若某个像素相加之和大于255，则相加后的值除以 255取余
        img = cv2.add(img1, img2)
    if use_numpy_or_opencv != 'numpy' and use_numpy_or_opencv != 'opencv':
        raise Exception("The Paragram Is Not In numpy And opencv!!")
    return img


if __name__ == '__main__':
    read_path = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "lena.jpg")
    print("read_path = {}".format(read_path))
    img1 = cv2.imread(read_path)
    gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = img1
    print("=============图像加法==============")
    plusImg = imgAdd(img1, img2, use_numpy_or_opencv='numpy')
    showImg(plusImg)
