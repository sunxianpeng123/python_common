# encoding: utf-8

"""
@author: sunxianpeng
@file: getImgPart.py
@time: 2020/7/21 11:50
"""


import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    read_path = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "lena.jpg")
    print("read_path = {}".format(read_path))
    img = readImg(read_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    print("*****************bgr***************")
    # bgr图像
    print("=======根据下标的方式取roi========")
    face = np.ones((180, 100, 3))
    face = img[220:400, 250:350]

    showImg(face)
    showImg(face)
    img[0:180, 0:100] = face
    showImg(img)