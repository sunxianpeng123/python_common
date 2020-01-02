# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_meshgrid.py
@time: 2019/12/31 20:52
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def tf_meshgrid():
    """ tf.meshgrid()用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的，它也可以是更高维的。"""

    print("######################1、tf_meshgrid #######################")
    print("使用numpy生成二维网格数据==============")
    # 无法使用gpu加速
    points_np = []
    for y in np.linspace(-2, 2, 5):
        for x in np.linspace(-2, 2, 5):
            points_np.append([x,y])
    points_np = np.array(points_np)
    # print(points_np)
    print("使用numpy生成二维网格数据==============")
    y = tf.linspace(-2., 2, 5)
    x = tf.linspace(-2., 2, 5)
    # 使用 tf.meshgrid 方法，产生两个指定形状的矩阵，空间中的一个点的坐标，
    # 由 tf.meshgrid 生成的所有矩阵中相同位置的数值组成
    points_tf_x, points_tf_y = tf.meshgrid(x, y)
    print('points_tf_x = {}'.format(points_tf_x))
    print('points_tf_y = {}'.format(points_tf_y))
    # points_tf_x = [[-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]]

    # points_tf_y = [[-2. -2. -2. -2. -2.]
    #  [-1. -1. -1. -1. -1.]
    #  [ 0.  0.  0.  0.  0.]
    #  [ 1.  1.  1.  1.  1.]
    #  [ 2.  2.  2.  2.  2.]]
    # 将空间中点的坐标合并到一个tensor中,需要添加一个维度
    points_tf = tf.stack([points_tf_x,points_tf_y],axis=2)# shape=(5, 5, 2)
    # print(points_tf)

    return None


def plot_contour():

    print("######################2、plot_contour #######################")
    x = tf.linspace(0., 2 * 3.14, 500)
    y = tf.linspace(0., 2 * 3.14, 500)
    # [50, 50]
    point_x, point_y = tf.meshgrid(x, y)
    # [50, 50, 2]
    points = tf.stack([point_x, point_y], axis=2)
    # points = tf.reshape(points, [-1, 2])
    print('points:', points.shape)

    z = tf.math.sin(points[..., 0]) + tf.math.sin(points[..., 1])
    print('z:', z.shape)

    plt.figure('plot 2d func value')
    plt.imshow(z, origin='lower', interpolation='none')
    plt.colorbar()

    plt.figure('plot 2d func contour')
    plt.contour(point_x, point_y, z)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    tf_meshgrid()
    plot_contour()