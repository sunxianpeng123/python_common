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
def imgTypeTransform(img1, img2):
    """
    图像加法
    :param img1:
    :param img2:
    :param
    OpenCV转换成PIL.Image格式
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    Image转换成OpenCV格式
        img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
    :return:
    """
    img = cv2.addWeighted(img1, 0.7, img2, 0.2, 0)
    return img

if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    lena = readImg(read_path_lena,False)
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)
    print("=================================")
    print("lena.shape = {}".format(lena.shape))
    # bgr 2 gray
    gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    # bgr 2 rgb
    rgb = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    # gray 2 bgr
    bgr1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    print(gray.shape)  # (512, 512)
    print(bgr1.shape)  # (512, 512, 3)
    # showImg(gray)
    # showImg(bgr1)


