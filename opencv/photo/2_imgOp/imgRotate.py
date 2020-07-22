# encoding: utf-8

"""
@author: sunxianpeng
@file: imgRotate.py
@time: 2020/7/21 14:41
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
def imgRotate(img, flip_code):
    """
    图像翻转
    :param img1:
    :return:flip(src, flipCode[, dst])
                1 	水平翻转，指以Y轴为中间线，左右翻转
                0 	垂直翻转，指以X轴为中间线，上下翻转
                -1 	水平垂直翻转，指先以Y轴为中间线，左右翻转，再以X轴为中间线，上下翻转
    """
    if flip_code not in [1, 0, -1]: raise Exception("Flip Code Error !!!")
    img_res = cv2.flip(img, flip_code)
    return img_res


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    showImg(img)
    for flip_code in [1, 0, -1]:
        img_res = imgRotate(img, flip_code)
        showImg(img_res)
