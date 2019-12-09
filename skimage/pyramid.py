# -*- coding: utf-8 -*-
# @Time : 2019/12/10 1:39
# @Author : sxp
# @Email : 
# @File : pyramid.py
# @Project : python_common
import numpy as np
import matplotlib.pyplot as plt
from skimage import data,transform


if __name__ == '__main__':
    image = data.astronaut()  # 载入宇航员图片
    rows, cols, dim = image.shape  # 获取图片的行数，列数和通道数

    pyramid = tuple(transform.pyramid_gaussian(image, downscale=2))  # 产生高斯金字塔图像
    # 共生成了log(512)=9幅金字塔图像，加上原始图像共10幅，pyramid[0]-pyramid[9]
    print(len(pyramid))
    print(pyramid[0].dtype.name)

    composite_image = np.ones((rows, cols + cols // 2, 3), dtype=np.double)  # 生成背景
    composite_image[:rows, :cols, :] = pyramid[0]  # 融合原始图像

    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p  # 循环融合9幅金字塔图像
        i_row += n_rows

    plt.imshow(composite_image)
    plt.show()
