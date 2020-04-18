# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_where.py
@time: 2019/12/31 19:53
"""

import tensorflow as tf


def tf_where():
    """ 传入mask，得到为true的坐标，使用gather_nd获取数据"""
    print("#################1、tf_where###################")
    a = tf.random.normal([3, 3])
    mask = a > 0#大于0位True，小于0位False

    # 以第一个维度为标准，取出为True的数据，组成一个新的tensor
    b = tf.boolean_mask(a, mask)#取出mask为true的数据
    print('a = {}'.format(a))
    print('mask = {}'.format(mask))
    print('b = {}'.format(b))
    # a = [[-0.75007695  0.78106815 -0.01807645]
    #  [-1.3096018   1.0700579  -1.3463402 ]
    #  [-1.8076766  -0.3533887   0.20802023]]

    # mask = [[False  True False]
    #  [False  True False]
    #  [False False  True]]

    # b = [0.78106815 1.0700579  0.20802023]
    print("使用where 实现和b相同结果===========")
    indices = tf.where(mask)#得到为True的坐标
    c = tf.gather_nd(a, indices)#根据坐标获取数据

    print('indices = {}'.format(indices))
    print('c = {}'.format(c))
    # indices = [[0 1]
    #  [1 1]
    #  [2 2]]

    # c = [0.78106815 1.0700579  0.20802023]

    return None

def tf_where_2():
    """传入三个参数，根据mask，根据mask中的true和false选择数据，
    true时从第二个参数中选择数据，false从第三个参数选择数据。
    所需数据的坐标和true or false 在mask中的坐标一致"""
    print("#################1、tf_where_2###################")
    mask = tf.constant([ [True, True, False],
                         [True, False, False],
                         [True, True, False]])
    A = tf.ones([3, 3])
    B = tf.zeros([3, 3])

    C = tf.where(mask, A, B)

    print('C = {}'.format(C))
    # C = [[1. 1. 0.]
    #  [1. 0. 0.]
    #  [1. 1. 0.]]
    return None

if __name__ == '__main__':
    tf_where()
    tf_where_2()