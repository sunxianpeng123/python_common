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
# 高通滤波器
#       高通滤波器（HPF）是检测图像的某个区域，然后根据像素与周围像素的亮度差值来提升像素的亮度。
#
#用于：边缘提取与增强。
#
#注意：通过高通滤波器进行滤波后，再和原图像叠加，可以增强图像中灰度级变化较快的部分，即锐化。
# ==============================================================================
def imgNumpyHPF(gray):
    """
    傅里叶变换得到低频、高频信息，针对他们不同处理可以实现不同目的
    傅里叶变换是可逆的，可以恢复原图
    在频域对图像进行处理，会反映在傅里叶逆变换生成的图像上
    :param gray:
    :return:
    """
    """实现傅里叶变换"""
    # 对图像进行傅里叶变换
    f = np.fft.fft2(gray)
    # 将低频从左上角移动到中心
    fshift = np.fft.fftshift(f)
    # 重置 区间,映射到[0,255]之间，以便使用图像显示
    # result = 20 * np.log(np.abs(fshift))

    """高通滤波"""
    # 得到边缘图像
    rows, cols = gray.shape
    # 取图像中心点
    row_mid, col_mid = int(rows / 2), int(cols / 2)
    # 去掉低频区域
    fshift[row_mid - 30:row_mid + 30, col_mid - 30:col_mid + 30] = 0

    """实现逆傅里叶变换"""
    # 将低频从中心移动到左上角
    ishift = np.fft.ifftshift(fshift)
    # 返回一个复数数组
    iimg = np.fft.ifft2(ishift)
    # 将上述复数数组重置区间到[0,255],便于图像显示
    iimg = np.abs(iimg)

    # 原始灰度图
    plt.subplot(1, 2, 1), plt.imshow(gray, 'gray'), plt.axis('off'), plt.title('original')
    # 频谱图像
    plt.subplot(1, 2, 2), plt.imshow(iimg, 'gray'), plt.axis('off'), plt.title('result')
    plt.show()

if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 127, 255, 0)
    print("img1 shape = {}".format(img.shape))
    imgNumpyHPF(img_gray)

