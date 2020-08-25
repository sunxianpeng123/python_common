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
# 低通滤波器
#   低通滤波器是像素与周围像素的亮度差值小于一个特定值时，平滑该像素的亮度。
#
# 用于：去噪和模糊化。
#
# 注意：低通滤波器容许低频信号通过，但减弱频率高于截止频率的信号的通过。
# ==============================================================================
def imgOpencvLPF(gray):
    """
    调节掩膜区域的大小可以得到不同模糊程度的图像，越大则越清晰，越小则越模糊。
    :param gray:
    :return:
    """
    """实现傅里叶变换"""
    # 对图像进行傅里叶变换
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将低频从左上角移动到中心
    dshift = np.fft.fftshift(dft)
    # 重置 区间,映射到[0,255]之间，以便使用图像显示
    # result = 20 * np.log(cv2.magnitude(dshift[:,:,0],dshift[:,:,1]))

    """低通滤波"""
    rows, cols = gray.shape
    # 取图像中心点
    row_mid, col_mid = int(rows / 2), int(cols / 2)
    # 构造掩膜图像
    mask = np.zeros((rows, cols, 2), np.uint8)

    # 去掉低频区域
    mask[row_mid - 30:row_mid + 30, col_mid - 30:col_mid + 30] = 1
    # 将频谱图像和掩膜相乘,保留低频部分，高频部分变成0
    md = dshift * mask

    """实现逆傅里叶变换"""
    # 将低频从中心移动到左上角
    imd = np.fft.ifftshift(md)
    # 返回一个复数数组
    iimg = cv2.idft(imd)
    # 将上述复数数组重置区间到[0,255],便于图像显示
    iimg = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])

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
    imgOpencvLPF(img_gray)


