# encoding: utf-8

import cv2
import matplotlib.pyplot as plt
import copy

# encoding: utf-8
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img(name="test", img=None):
    plt.figure()
    plt.imshow(img)
    plt.title(name)
    plt.show()


if __name__ == '__main__':
    path = r"lena.jpg"
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, 0)
    """测试使用不同的cmap，画彩色和灰度图像"""
    # 使用cmap，默认值显示彩色图像，不正确
    plt.subplot(2, 2, 1), plt.imshow(img), plt.axis('off')  # 关闭坐标轴
    # 使用cmap='gray'显示彩色图像，不正确
    plt.subplot(2, 2, 2), plt.imshow(img, cmap='gray'), plt.axis('off')  # 关闭坐标轴
    # 使用cmap，默认值显示灰度图像，不正确
    plt.subplot(2, 2, 3), plt.imshow(gray), plt.axis('off')  # 关闭坐标轴
    # 使用cmap='gray',默认值显示彩色图像，正确
    plt.subplot(2, 2, 4), plt.imshow(gray, 'gray'), plt.axis('off')  # 关闭坐标轴

    plt.annotate('t')
    plt.show()