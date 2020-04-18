# encoding: utf-8

"""
@author: sunxianpeng
@file: 2_scatter_nd.py
@time: 2019/12/31 20:28
"""

import tensorflow as tf


def tf_scatter_nd_1_dim():
    """ 根据已有的坐标和数据，在指定的tensor上更新数据"""
    print("###################1、tf_scatter_nd_1_dim##################")
    indices = tf.constant([[4], [3], [1], [7]])#4*1
    update_data = tf.constant([9, 10, 11, 12])#tf.Tensor([ 9 10 11 12], shape=(4,), dtype=int32)
    shape = tf.constant([8])#长度为 8 的向量，每个位置元素均为 0

    print(indices)
    # tf.Tensor(
    # [[4]
    #  [3]
    #  [1]
    #  [7]], shape=(4, 1), dtype=int32)
    print(update_data)#tf.Tensor([ 9 10 11 12], shape=(4,), dtype=int32)
    print(shape)#tf.Tensor([8], shape=(1,), dtype=int32)
    # 根据indices 指定的下标索引（即shape中需要更新数据的位置）,依次将update_data中的数据更新在shape中
    r = tf.scatter_nd(indices, update_data, shape)
    print(r)#tf.Tensor([ 0 11  0 10  9  0  0 12], shape=(8,), dtype=int32)
    return None

def tf_scatter_nd_multi_dims():
    """ 根据已有的坐标和数据，在指定的tensor上更新数据"""
    print("###################1、tf_scatter_nd_2_dim##################")
    # [0]指的第一个维度上的第一个值（4*4）
    # [2]指的第一个维度上的第2个值（4*4）
    indices = tf.constant([[0], [2]])
    update = tf.constant([
        [[5, 5, 5, 5],
         [6, 6, 6, 6],
         [7, 7, 7, 7],
         [8, 8, 8, 8]],
        [[5, 5, 5, 5],
         [6, 6, 6, 6],
         [7, 7, 7, 7],
         [8, 8, 8, 8]],
    ])
    shape = tf.constant([4,4,4])#模板维度，元素均为0

    print(indices)
    # tf.Tensor(
    # [[0]
    #  [2]], shape=(2, 1), dtype=int32)
    print('update shape = {}'.format(update.shape))# update shape = (2, 4, 4)

    r = tf.scatter_nd(indices, update, shape)
    print(r)
    # tf.Tensor(
    # [[[5 5 5 5]
    #   [6 6 6 6]
    #   [7 7 7 7]
    #   [8 8 8 8]]
    #
    #  [[0 0 0 0]
    #   [0 0 0 0]
    #   [0 0 0 0]
    #   [0 0 0 0]]
    #
    #  [[5 5 5 5]
    #   [6 6 6 6]
    #   [7 7 7 7]
    #   [8 8 8 8]]
    #
    #  [[0 0 0 0]
    #   [0 0 0 0]
    #   [0 0 0 0]
    #   [0 0 0 0]]], shape=(4, 4, 4), dtype=int32)


    return None

if __name__ == '__main__':
    tf_scatter_nd_1_dim()
    tf_scatter_nd_multi_dims()