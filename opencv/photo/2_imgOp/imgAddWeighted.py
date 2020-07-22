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
def imgAddWeighted(img1, img2):
    """
    图像加法
    :param img1:
    :param img2:
    :param use_numpy_or_opencv:使用numpy做图像加法还是使用opencv做图像加法
    :return:addWeighted（）函数是将两张相同大小，相同类型的图片融合的函数。他可以实现图片的特效。
        src1：插入的第一个图片；
        src2：插入的第二个图片；
        alpha：double类型，加权系数，是src1图片的融合占比 ；
        beta：double类型，加权系数，是src2图片的融合占比；
        gamma：亮度调节值；
    该函数的计算公式为:
        dst(I)=saturate(src1(I)∗alpha+src2(I)∗beta+gamma)
    """
    img = cv2.addWeighted(img1, 0.7, img2, 0.2, 0)

    return img


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "lena.jpg")
    read_path_scenery = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "scenery.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    print("read_path_scenery = {}".format(read_path_scenery))

    lena = readImg(read_path_lena,False)
    scenery = readImg(read_path_scenery,False)
    scenery = cv2.resize(scenery, (512, 512), interpolation=cv2.INTER_CUBIC)

    lena_gray = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)
    scenery_gray = cv2.cvtColor(lena, cv2.COLOR_RGB2GRAY)
    print("=================================")
    print("lena.shape = {}".format(lena.shape))
    print("scenery.shape = {}".format(scenery.shape))
    img_result = cv2.addWeighted(lena, 0.7, scenery, 0.2, 0)
    showImg(img_result)
