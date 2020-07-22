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


def saveImg(img, save_path, path_has_chinese=True):
    """
    保存图片
    :param save_path:保存路径
    :param path_has_chinese: 路径中是否有中文，默认有中文
    :return:
    """
    if not path_has_chinese:
        cv2.imwrite(save_path, img)
    else:
        # # opencv 通道顺序 brg，PIL 通道顺序 RGB
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save(save_path)


if __name__ == '__main__':
    read_path = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"), "lena.jpg")
    save_path = os.path.join(os.path.abspath("./"), "saved.jpg")
    print("read_path = {}".format(read_path))
    print("save_path = {}".format(save_path))

    img = readImg(read_path, path_has_chinese=True)
    showImg(img)
    saveImg(img, save_path, path_has_chinese=True)
