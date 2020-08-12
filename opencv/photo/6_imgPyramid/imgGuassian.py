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
# 1、对图像的向下取样操作，即缩小图像。
#     Gaussian金字塔是是通过依次地向下迭代采样（从底部到顶部）获得整个金字塔，随着依次地采样，图像越来越小。
#     第（i+1）层 Gi+1，是由第 i 层 Gi  和高斯核进行卷积，然后去除每个偶数行和列，得到的采样图像是前一层的（1/4）。
#     由其实现过程可知，向下采样是有损的操作，会丢弃了部分信息；
#     图像的层次越高，对应的图像越小，分辨率也越低。
# 2、对图像的向上取样，即放大图像
#     将图像在每个方向扩大为原来的两倍，新增的行和列以0填充
#     使用先前同样的内核(乘以4)与放大后的图像卷积，获得 “新增像素”的近似值
#     得到的图像即为放大后的图像，但是与原来的图像相比会发觉比较模糊，因为在缩放的过程中已经丢失了一些信息，如果想在缩小和放大整个过程中减少信息的丢失，这些数据形成了拉普拉斯金字塔。
# ==============================================================================
def imgGuassianPyrDown(img):
    """
    向下采样，图片每一次采样，图片缩小 1/4,即横纵都缩小 1/2
    opencv的pyrDown函数先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）。
    其函数原型为：
        pyrDown(src[, dst[, dstsize[, borderType]]]) -> dst
            src参数表示输入图像。
            dst参数表示输出图像，它与src类型、大小相同。
            dstsize参数表示降采样之后的目标图像的大小。
                    它是有默认值的，如果我们调用函数的时候不指定第三个参数，那么这个值是按照 Size((src.cols+1)/2, (src.rows+1)/2) 计算的。
                    而且不管你自己如何指定这个参数，一定必须保证满足以下关系式：
                            |dstsize.width * 2 - src.cols| ≤ 2;
                            |dstsize.height * 2 - src.rows| ≤ 2。
                    也就是说降采样的意思其实是把图像的尺寸缩减一半，行和列同时缩减一半。
            borderType参数表示表示图像边界的处理方式。
    :param gray:
    :return:
    """
    # 1
    # 向下采样，图片每一次采样，图片缩小 1/4,即横纵都缩小 1/2
    pyr_down_1 = cv2.pyrDown(img)
    pyr_down_2 = cv2.pyrDown(pyr_down_1)
    pyr_down_3 = cv2.pyrDown(pyr_down_2)
    print("img shape = {}\npyrDown_1 shape = {}\npyrDown_2 shape = {}\npyrDown_3 shape = {} "
          .format(img.shape, pyr_down_1.shape, pyr_down_2.shape, pyr_down_3.shape))
    showImgs([img, pyr_down_1, pyr_down_2, pyr_down_3])


def imgGuassianPyrUp(img):
    """
        opencv的pyrUp函数先对图像进行升采样（将图像尺寸行和列方向增大一倍），然后再进行高斯平滑。
        其函数原型为：
            pyrUp(src[, dst[, dstsize[, borderType]]]) -> dst
                src参数表示输入图像。
                dst参数表示输出图像，它与src类型、大小相同。
                dstsize参数表示降采样之后的目标图像的大小。在默认的情况下，这个尺寸大小是按照 Size(src.cols*2, (src.rows*2) 来计算的。
                    如果你自己要指定大小，那么一定要满足下面的条件：
                        |dstsize.width - src.cols * 2| ≤ (dstsize.width mod 2);  //如果width是偶数，那么必须dstsize.width是src.cols的2倍
                        |dstsize.height - src.rows * 2| ≤ (dstsize.height mod 2);
                borderType参数表示表示图像边界的处理方式。

    :param gray:
    :return:
    """
    # 1
    # 向下采样，图片每一次采样，图片缩小 1/4,即横纵都缩小 1/2
    # 向上采样，图片每一次采样，图片放大4倍,即横纵都放大 2 倍
    pyr_up_1 = cv2.pyrUp(img)
    pyr_up_2 = cv2.pyrUp(pyr_up_1)
    pyr_up_3 = cv2.pyrUp(pyr_up_2)

    print("img shape = {}\npyr_up_1 shape = {}\npyr_up_2 shape = {}\npyr_up_3 shape = {} "
          .format(img.shape, pyr_up_1.shape, pyr_up_2.shape, pyr_up_3.shape))
    showImgs([img, pyr_up_1, pyr_up_2, pyr_up_3])


def imgGuassianPyrRecover(img):
    """
    向上采样和向下采样的逆操作
    :param gray:
    :return:
    """
    # 1
    # 向下采样，图片每一次采样，图片缩小 1/4,即横纵都缩小 1/2
    # 向上采样，图片每一次采样，图片放大4倍,即横纵都放大 2 倍
    # 1
    # 向下采样，图片每一次采样，图片缩小 1/4,即横纵都缩小 1/2
    # 向上采样，图片每一次采样，图片放大4倍,即横纵都放大 2 倍
    # 第一次 先向上取样，再向下采样
    pyr_down_1 = cv2.pyrDown(img)
    pyr_up_1 = cv2.pyrUp(pyr_down_1)
    # 第二次 先向上取样，再向下采样
    pyr_down_2 = cv2.pyrDown(pyr_up_1)
    pyr_up_2 = cv2.pyrUp(pyr_down_2)
    # 第三次 先向上取样，再向下采样
    pyr_down_3 = cv2.pyrDown(pyr_up_2)
    pyr_up_3 = cv2.pyrUp(pyr_down_3)

    # 查看两幅图的像素差异
    diff = pyr_up_3 - img
    showImgs([diff, pyr_up_3])
    # print("img shape = {}\npyrDown_1 shape = {}\npyrDown_2 shape = {}\npyrDown_3 shape = {} "
    #       .format(img.shape, pyr_down_1.shape,pyr_down_2.shape, pyr_down_3.shape))
    print("img shape = {}\npyr_up_1 shape = {}\npyr_up_2 shape = {}\npyr_up_3 shape = {} "
          .format(img.shape, pyr_up_1.shape, pyr_up_2.shape, pyr_up_3.shape))


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print("img1 shape = {}".format(img.shape))

    """0 == imgGuassianPyrDown,,,1 == imgGuassianPyrUp,,,2 == imgGuassianPyrUp"""
    runType = 2
    if runType == 0:
        print("================向下采样=================")
        # 向下采样，图片每一次采样，图片缩小 1/4,即横纵都缩小 1/2
        imgGuassianPyrDown(img)
    elif runType == 1:
        print("================向上采样=================")
        # 向上采样，图片每一次采样，图片放大4倍,即横纵都放大 2 倍
        imgGuassianPyrUp(img)
    elif runType == 2:
        print("================向上采样和向下采样的逆操作=================")
        imgGuassianPyrRecover(img)
    else:
        raise Exception("The Value Of runType Should In （0,1,2）")
