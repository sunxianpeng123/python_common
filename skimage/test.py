# -*- coding: utf-8 -*-
# @Time : 2019/12/10 0:37
# @Author : sxp
# @Email : 
# @File : data_type.py
# @Project : python_common

from skimage import io,transform,data
import numpy as np
import matplotlib.pylab as plt

def show_img(img):
    io.imshow(img)
    io.show()


if __name__ == '__main__':
    img = data.camera()
    dst = transform.resize(img,(80,60))

    print(img.shape)  # 图片原始大小
    img1 = transform.rotate(img, 90)  # 旋转90度，不改变大小
    print(img1.shape)
    img2 = transform.rotate(img, 30, resize=True)  # 旋转30度，同时改变大小
    print(img2.shape)

    plt.figure('resize')

    plt.subplot(121)
    plt.title('rotate 90')
    plt.imshow(img1, plt.cm.gray)

    plt.subplot(122)
    plt.title('rotate  30')
    plt.imshow(img2, plt.cm.gray)

    plt.show()