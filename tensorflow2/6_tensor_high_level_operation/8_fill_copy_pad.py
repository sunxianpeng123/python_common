# encoding: utf-8

"""
@author: sunxianpeng
@file: 8_fill_and_copy.py
@time: 2019/12/30 18:59
"""

import tensorflow as tf

def tf_pad():
    """数据填充 """
    print("################1、tf_pad 填充 0######################")
    print("一维数据=============")
    a = tf.constant([1,2,3])
    # [[1,2]] 表示 在左边填充1个0，右边填充2个0
    b = tf.pad(a, [[1, 2]])

    print('a = {}'.format(a))
    print('b = {}'.format(b))
    # a = [1 2 3]
    # b = [0 1 2 3 0 0]
    print("二维数据=============")
    c = tf.reshape(tf.range(9), [3,3])
    # [[0, 0], [0, 0]]是一个二维矩阵，表示，
    # 第一个 [0, 0]表示第一维，在行上看：上方和下方补充 0行
    # 第二个 [0, 0]表示第二维，在列上看：左侧和右侧补充 0列
    d = tf.pad(c, [[0, 0], [0, 0]])
    # [[1, 0], [1, 0]]是一个二维矩阵，表示，
    # 第一个 [1, 0]表示第一维，在行上看：上方补充1行，下方补充0行
    # 第二个 [1, 0]表示第二维，在列上看：左侧补充1列，右侧补充0列
    e = tf.pad(c, [[2, 0], [1, 0]])

    print('c = {}'.format(c))
    print('d = {}'.format(d))
    print('e = {}'.format(e))
    # c = [[0 1 2]
    #  [3 4 5]
    #  [6 7 8]]

    # d = [[0 1 2]
    #  [3 4 5]
    #  [6 7 8]]

    # e = [[0 0 0 0]
    #  [0 0 0 0]
    #  [0 0 1 2]
    #  [0 3 4 5]
    #  [0 6 7 8]]
    return None

def tf_image_pad():
    """ 在图像周围天填充数据"""
    print("################2、tf_image_pad ######################")
    a = tf.random.normal([4, 28, 28, 3])
    # 在图像的上方和下方都添加两行,在图像的左侧和右侧都添加两列
    # batch 和 channel 维度不填充数据
    b = tf.pad(a,[[0, 0], [2,2], [2,2], [0, 0]])

    print('a shape = {}'.format(a.shape))
    print('b shape = {}'.format(b.shape))
    return None

def tf_tile():
    """数据复制 tf.tile(input, multiples, name=None)
            input是待扩展的张量，
            multiples是扩展方法。
    """
    print("################3、tf_tile ######################")
    a = tf.reshape(tf.range(9), [3, 3])
    # [1, 2] 表示经过tile复制后
    # 第一个维度是复制前的1倍（即保持不变），第二个维度是复制前的两倍
    # 即行数不变，列数变为之前的两倍
    b = tf.tile(a,[1, 2])
    # [2, 1] 表示经过tile复制后
    # 第一个维度是复制前的2倍，第二个维度是复制前的1倍(即保持不变）
    # 即行数变为之前的两倍，列数不变
    c = tf.tile(a,[2, 1])
    # [2, 2] 表示经过tile复制后
    # 第一个维度是复制前的2倍，第二个维度是复制前的两倍
    # 即行数变为之前的两倍，列数变为之前的两倍
    d = tf.tile(a, [2, 2])

    print('a shape = {}'.format(a.shape))
    print('b shape = {}'.format(b.shape))
    print('c shape = {}'.format(c.shape))
    print('d shape = {}'.format(d.shape))

    return None

def tf_boradcastto_vs_tile():
    """参考4_broadcast"""
    return None

if __name__ == '__main__':
    tf_pad()
    tf_image_pad()
    tf_tile()