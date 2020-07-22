# encoding: utf-8

"""
@author: sunxianpeng
@file: handlePixel.py
@time: 2020/7/21 11:39
"""
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

"""
    在彩色图像中，图像可用一个二维矩阵表示，其中每个元素是RGB值，RGB每个占一个字节（8位），总共三个字节，每个字节取值范围是0-2^8(255)，
    也可以说是三通道的，分别是R,G,B，比如（255，0，0）代表红色，(0,255,0)代表绿色，(0,0,255)代表蓝色，其他颜色是这三个颜色的混合。
 
    在灰色图像中，每个元素只有单通道，即只有一个值。  
    
    位深度：位深度是把通道数转化为位，即3通道位深度为3*8=24位。 
     
    分辨率：分辨率即为高像素*宽像素，如1080*1440分辨率 
      
DPI（Dots Per Inch，每英寸像素点数)：   
    通过DPI和像素可以求出图片的实际尺寸，如1080*1440分辨率，DPI96，那么图片实际高为1080/96 = 11.25英寸，宽为1440/96=15英寸。
    图片清晰程度不是由像素决定，而是用DPI来决定，DPI越大，清晰度越高。
注：
    鼠标的DPI参数指的是鼠标在桌面上移动1英寸的距离的同时，鼠标光标能够在屏幕上移动多少“点”。越高，移动越快越灵敏。
    
    OpenCV中图像的三原色顺序为 BGR

"""


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
    # bgr图像
    print("*****************1、bgr三通道图像***************")
    # 1、修改某个点上的三个通道的值
    b = img[78, 125, 0]
    g = img[78, 125, 1]
    r = img[78, 125, 2]
    img[78, 125, 0] = 100  # 赋值,修改值
    bgr = img[78, 125]
    print("b={}，g={}，r={}，bgr={}".format(b, g, r, bgr))
    # 2
    print("img.item(100, 100, 0) = {}".format(img.item(100, 100, 0)))  # 查看值
    img.itemset((100, 100, 0), 255)  # 修改值
    print("img.item(100, 100, 0) = {}".format(img.item(100, 100, 0)))

    print("========修改多个行列数值=========")
    # 1、修改某个区域的像素值
    i1 = copy.deepcopy(img)
    img[100:150, 100:150] = [255, 255, 255]
    showImgs([i1, img])
    # 灰度图
    print("*****************2、gray灰度图像***************")
    p = gray[78, 125]
    print("p={}".format(p))
    gray[78, 125] = 125  # 赋值
    print("p={}".format(p))
    # 2
    print(gray.item(100, 100))  # 查看值
    gray.itemset((100, 100), 255)  # 修改值
    print(gray.item(100, 100))
