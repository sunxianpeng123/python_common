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
# 运用它，首先就要了解它，什么是平滑滤波？
#       平滑滤波是低频增强的空间域滤波技术。它的目的有两类：
#       一类是模糊；另一类是消除噪音。
#       空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。
# 在看一下滤波的目的：
# 滤波的本义是指信号有各种频率的成分,滤掉不想要的成分,即为滤掉常说的噪声,留下想要的成分.这即是滤波的过程,也是目的。
#     抽出对象的特征作为图像识别的特征模式；
#     为适应图像处理的要求，消除图像数字化时所混入的噪声。
# ==============================================================================
def imgGaussianFilter(gray):
    """
    高斯滤波器
    :param img:
    :return:现在把卷积模板中的值换一下，不是全1了，换成一组符合高斯分布的数值放在模板里面，比如这时中间的数值最大，往两边走越来越小，构造一个小的高斯包。实现的函数为cv2.GaussianBlur()。
            对于高斯模板，我们需要制定的是高斯核的高和宽（奇数），沿 x 与 y 方向的标准差(如果只给x，y=x，如果都给0，那么函数会自己计算)。高斯核可以有效的出去图像的高斯噪声。当然也可以自己构造高斯核
            cv2.Guassianblur(img, (3, 3), 1) 表示进行高斯滤波，
        参数说明:
            1表示σ， x表示与当前值得距离，计算出的G(x)表示权重值
        卷积核大小是奇数
        dst = cv2.GaussianBlur(src,ksize,sigmaX,sigmay,borderType)
            src: 输入图像矩阵,可为单通道或多通道，多通道时分别对每个通道进行卷积
            dst:输出图像矩阵,大小和数据类型都与src相同
            ksize:高斯卷积核的大小，宽，高都为奇数，且可以不相同
            sigmaX: 一维水平方向高斯卷积核的标准差
            sigmaY: 一维垂直方向高斯卷积核的标准差，默认值为0，表示与sigmaX相同
            borderType:填充边界类型
    """
    # 添加点噪声
    for i in range(2000):
        temp_x = np.random.randint(0, gray.shape[0])
        temp_y = np.random.randint(0, gray.shape[1])
        gray[temp_x][temp_y] = 255

    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(gray, 'gray')  # 默认彩色，指定为gray
    plt.subplot(1, 2, 2), plt.imshow(gauss, 'gray')

    plt.show()



if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))
    print("=================================")
    imgGaussianFilter(img_gray)
