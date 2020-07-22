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
# 处理灰度图
#   一幅图像包括目标物体、背景还有噪声，要想从多值的数字图像中直接提取出目标物体，常用的方法就是设定一个阈值 T，用 T 将图像的数据分成两部分：大于T的像素群和小于T的像素群。这是研究灰度变换的最特殊的方法，称为图像的二值化（Binarization）。
#   阈值分割法的特点是:适用于目标与背景灰度有较强对比的情况，重要的是背景或物体的灰度比较单一，而且总可以得到封闭且连通区域的边界。

# ==============================================================================
def imgSimpleBinary(gray):
    """
    简单阈值分割
    :param img:
    :return:选取一个全局阈值，然后就把整幅图像分成非黑即白的二值图像。cv2.threshold( )
        这个函数有四个参数，第一个是原图像矩阵，第二个是进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数，常用的有：
            cv2.THRESH_BINARY（黑白二值）
            cv2.THRESH_BINARY_INV（黑白二值翻转）
            cv2.THRESH_TRUNC（得到额图像为多像素值）
            cv2.THRESH_TOZERO（当像素高于阈值时像素设置为自己提供的像素值，低于阈值时不作处理）
            cv2.THRESH_TOZERO_INV（当像素低于阈值时设置为自己提供的像素值，高于阈值时不作处理）
    """
    # binary （黑白二值）0,255
    ret1, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # （黑白二值反转）255,0
    ret2, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # 得到的图像为多像素值
    ret3, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    # 高于阈值时像素设置为255，低于阈值时不作处理
    ret4, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    # 低于阈值时设置为255，高于阈值时不作处理
    ret5, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)
    showImg(thresh1)

def imgAaptiveBinary(gray):
    """
    自适应阈值分割
    :param gray:
    :return:一中的简单阈值是一种全局性的阈值，只需要设定一个阈值，整个图像都和这个阈值比较。而自适应阈值可以看成一种局部性的阈值，
        通过设定一个区域大小，比较这个点与区域大小里面像素点 的平均值（或者其他特征）的大小关系确定这个像素点的情况。使用的函数为：cv2.adaptiveThreshold()
        参数：
            第一个参数 src     指原图像，原图像应该是灰度图。
            第二个参数 x       指当像素值高于（有时是小于）阈值时应该被赋予的新的像素值
            第三个参数 adaptive_method  指： CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C
            第四个参数  threshold_type  指取阈值类型：必须是下者之一 CV_THRESH_BINARY，CV_THRESH_BINARY_INV
            第五个参数 block_size  指用来计算阈值的象素邻域大小: 3, 5, 7, ...
            第六个参数 param1   指与方法有关的参数。对方法CV_ADAPTIVE_THRESH_MEAN_C 和 CV_ADAPTIVE_THRESH_GAUSSIAN_C， 它是一个从均值或加权均值提取的常数, 尽管它可以是负数。
        自适应阈值：
            对方法CV_ADAPTIVE_THRESH_MEAN_C，先求出块中的均值，再减掉param1。
            对方法 CV_ADAPTIVE_THRESH_GAUSSIAN_C ，先求出块中的加权和(gaussian)， 再减掉param1。
    """
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # 第一个参数为原始图像矩阵，第二个参数为像素值上限，第三个是自适应方法（adaptive method）：
    #           -----cv2.ADAPTIVE_THRESH_MEAN_C:领域内均值
    #           -----cv2.ADAPTIVE_THRESH_GAUSSIAN_C:领域内像素点加权和，权重为一个高斯窗口
    # 第四个值的赋值方法：只有cv2.THRESH_BINARY和cv2.THRESH_BINARY_INV
    # 第五个Block size：设定领域大小（一个正方形的领域）
    # 第六个参数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值，就是求得领域内均值或者加权值）
    # 这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图都用一个阈值
    th2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def imgOtsusBinary(gray):
    """

    :param gray:
    :return:
    我们前面说到，cv2.threshold函数是有两个返回值的，前面一直用的第二个返回值，也就是阈值处理后的图像，那么第一个返回值（得到图像的阈值）将会在这里用到。

    前面对于阈值的处理上，我们选择的阈值都是127，那么实际情况下，怎么去选择这个127呢？有的图像可能阈值不是127得到的效果更好。那么这里我们需要算法自己去寻找到一个阈值，
    而Otsu’s就可以自己找到一个认为最好的阈值。并且Otsu’s非常适合于图像灰度直方图具有双峰的情况，他会在双峰之间找到一个值作为阈值，对于非双峰图像，可能并不是很好用。

    那么经过Otsu’s得到的那个阈值就是函数cv2.threshold的第一个参数了。因为Otsu’s方法会产生一个阈值，那么函数cv2.threshold的的第二个参数（设置阈值）就是0了，并且在cv2.threshold的方法参数中还得加上语句cv2.THRESH_OTSU
    那么什么是双峰图像（只能是灰度图像才有），就是图像的灰度统计图中可以明显看出只有两个波峰，比如下面一个图的灰度直方图就可以是双峰图：
    """
    # 简单滤波
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Otsu 滤波
    ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.figure()
    plt.subplot(221), plt.imshow(gray, 'gray')
    # .ravel方法将矩阵转化为一维,画出灰度直方图
    plt.subplot(222), plt.hist(gray.ravel(), 256)
    plt.subplot(223), plt.imshow(th1, 'gray')
    plt.subplot(224), plt.imshow(th2, 'gray')
    plt.show()

if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgSimpleBinary(img_gray)
    imgAaptiveBinary(img_gray)
    imgOtsusBinary(img_gray)


