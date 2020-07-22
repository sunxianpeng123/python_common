# encoding: utf-8

"""
@author: sunxianpeng
@file: splitImg.py
@time: 2020/7/21 11:56
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
    print("===============通道拆分==================")
    # 1第一种拆分方法
    b1 = img[:, :, 0]
    g1 = img[:, :, 1]
    r1 = img[:, :, 2]
    # 2 第二种拆分方法
    b2, g2, r2 = cv2.split(img)
    b3 = cv2.split(img)[0]
    g3 = cv2.split(img)[1]
    r3 = cv2.split(img)[2]
    # showImg(b2)
    print("=============通道合并，注意顺序==============")
    # 1 合并通道，组成原图
    m = cv2.merge([b1, g1, r1])
    # showImg(m)
    # 2
    bt = cv2.split(img)[0]
    rows, cols, chn = img.shape
    gt = np.zeros((rows, cols), dtype=img.dtype)
    rt = np.zeros((rows, cols), dtype=img.dtype)
    mt = cv2.merge([bt, gt, rt])
    showImg(mt)
