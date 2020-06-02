# encoding: utf-8

"""
@author: sunxianpeng
@file: 4_reduce_min_max_mean.py
@time: 2019/12/29 20:21
"""


import tensorflow as tf

def tf_reduce_min_max_mean_sum():
    """求解tensor的最大值最小值，因为会有降维过程，所以有前缀reduce"""
    print("###################1、tf_reduce_min_max_mean_sum ####################")
    print("求解整个tensor的各种统计值================")
    t = tf.random.normal([2,3])
    t_1 = tf.reduce_min(t)
    t_2 = tf.reduce_max(t)
    t_3 = tf.reduce_mean(t)
    t_4 = tf.reduce_sum(t)

    print('t = {}'.format(t))
    print('t_1 = {}'.format(t_1))
    print('t_1 shape = {}'.format(t_1.shape))#标量，形状 （）
    print('t_2 = {}'.format(t_2))#
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 = {}'.format(t_3))
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_4 = {}'.format(t_4))
    print('t_4 shape = {}'.format(t_4.shape))
    # t = [[ 0.56751156 -1.923078    0.18067797]
    #  [ 0.06871225  1.3103118  -0.07443979]]

    # t_1 = -2.6603376865386963
    # t_1 shape = ()
    # t_2 = 2.343855142593384
    # t_2 shape = ()
    # t_3 = -0.005003327038139105
    # t_3 shape = ()
    # t_4 = -0.2001330852508545
    # t_4 shape = ()
    print("求解指定维度的统计值======================")
    # axis 表示下标索引
    # axis=1 表示以第2个维度上的shape分组，即每组数据有三个
    # t = tf.random.normal([2,3])
    print("t.shape = {}".format(t.shape))
    t_5 = tf.reduce_min(t, axis=1)#
    t_6 = tf.reduce_max(t, axis=1)
    t_7 = tf.reduce_mean(t, axis=1)
    t_8 = tf.reduce_sum(t, axis=1)
    print('t = {}'.format(t))
    print('t_5 = {}'.format(t_5))
    print('t_5 shape = {}'.format(t_5.shape))
    print('t_6 = {}'.format(t_6))#
    print('t_6 shape = {}'.format(t_6.shape))
    print('t_7 = {}'.format(t_7))
    print('t_7 shape = {}'.format(t_7.shape))
    print('t_8 = {}'.format(t_8))
    print('t_8 shape = {}'.format(t_8.shape))
    # t = [[-0.46380982  0.14615561  0.8601456 ]
    #  [ 1.2590717  -0.45682818  0.56364626]]

    # t_5 = [-0.46380982 -0.45682818]
    # t_5 shape = (2,)
    # t_6 = [0.8601456 1.2590717]
    # t_6 shape = (2,)
    # t_7 = [0.18083048 0.4552966 ]
    # t_7 shape = (2,)
    # t_8 = [0.54249144 1.3658898 ]
    # t_8 shape = (2,)
    return None

if __name__ == '__main__':
    tf_reduce_min_max_mean_sum()