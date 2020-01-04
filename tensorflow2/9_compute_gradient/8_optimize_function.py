# -*- coding: utf-8 -*-
# @Time : 2020/1/4 23:09
# @Author : sxp
# @Email : 
# @File : 8_optimize_function.py
# @Project : python_common

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def himmelblau(x):
    """函数"""
    return (x[0]**2 + x[1] - 11) **2 + (x[0] + x[1]**2 - 7)**2

def plot_func_picture():
    """画出函数图像"""
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    print('x,y range:', x.shape, y.shape)
    X, Y = np.meshgrid(x, y)
    print('X,Y maps:', X.shape, Y.shape)
    Z = himmelblau([X, Y])

    fig = plt.figure('himmelblau')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

def find_better_result():
    x = tf.constant([4., 0.])
    for step in range(200):
        with tf.GradientTape() as tape:
            # 若x为Variable类型，则不需要watch
            tape.watch(x)
            y = himmelblau(x)
        # 将所求梯度从list中取出来
        grads = tape.gradient(y, [x])[0]
        x -= 0.01 * grads
        if step % 20 ==0:
            print('step {}:x = {},f(x) = {}'.format(step,x.numpy(), y.numpy()))


if __name__ == '__main__':
    """求解函数
    f(x,y)= (x**2 + y - 11) ** + (x + y**2 - 7)**2
    的最小值
    """
    # plot_func_picture()
    find_better_result()




