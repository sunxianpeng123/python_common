# -*- coding: utf-8 -*-
# @Time : 2019/12/12 0:09
# @Author : sxp
# @Email : 
# @File : autolevel.py
# @Project : python_common

import matplotlib.pyplot as plt
from skimage import io,data,color
from skimage.morphology import disk
import skimage.filters.rank as sfr



if __name__ == '__main__':
    path = r"E:\PythonProjects\python_common\skimage\images\lena.jpg"
    img = io.imread(path)
    img = color.rgb2gray(img)
    auto = sfr.enhance_contrast(img,disk(5))#半径为5的圆形滤波器

    plt.figure('filters', figsize=(8, 8))
    plt.subplot(121)
    plt.title('origin image')
    plt.imshow(img, plt.cm.gray)

    plt.subplot(122)
    plt.title('filted image')
    plt.imshow(auto, plt.cm.gray)
    plt.show()
