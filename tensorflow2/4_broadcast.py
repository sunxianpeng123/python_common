# -*- coding: utf-8 -*-
# @Time : 2019/12/27 1:10
# @Author : sxp
# @Email : 
# @File : 4_broadcast.py
# @Project : python_common

import tensorflow as tf

def tf_broadcast():
    """
    张量维度扩张的手段，但是没有在数据层面上的复制。是一种数据优化的手段。高效且直观。
    原理参考
    https://blog.csdn.net/z_feng12489/article/details/89332012
    """
    print("###############1、tf_broadcast################")
    # x*w + b
    t = tf.random.normal([4,32,32,3])

    b_1 = tf.random.normal([3])
    b_2 = tf.random.normal([32,32,1])
    b_3 = tf.random.normal([4,1,1,1])
    b_4 = tf.random.normal([1,4,1,1])

    """先扩展成相同的维度个数，再执行计算"""
    # b_1自动扩张成[4,32,32,3],流程[3] ->[1,1,1,3] ->[4,32,32,3]
    t_1 = t + b_1
    # b_2自动扩张成[4,32,32,3],流程[32,32,1] ->[1,32,32,1] ->[4,32,32,3]
    t_2 = t + b_2
    # b_3自动扩张成[4,32,32,3],流程[4,1,1,1] ->[4,1,1,1] ->[4,32,32,3]
    t_3 = t + b_3
    # 因为b_4和t的第二个维度处不相同，无法扩张，计算出错
    # t_4 = t + b_4
    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    # print('t_4 shape = {}'.format(t_4.shape))

    # t shape = (4, 32, 32, 3)
    # t_1 shape = (4, 32, 32, 3)
    # t_2 shape = (4, 32, 32, 3)
    # t_3 shape = (4, 32, 32, 3)
    return None

def tf_broadcast_to():
    """ 原理参考
    https://blog.csdn.net/z_feng12489/article/details/89332012
    """
    print("###############2、tf_broadcast_to################")
    t = tf.random.normal([4,32,32,3])

    b_1 = tf.random.normal([4,1,1,1])
    # 将b_1主动扩张成[4,32,32,3]
    b_2 = tf.broadcast_to(b_1,[4,32,32,3])

    # b_3自动扩张成[4,32,32,3],流程[4,1,1,1] ->[4,1,1,1] ->[4,32,32,3]
    t_1 = t + b_1
    # 结果和t_1相同
    t_2 = t + b_2

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('b_1 shape = {}'.format(b_1.shape))
    print('b_2 shape = {}'.format(b_2.shape))
    # t shape = (4, 32, 32, 3)
    # t_1 shape = (4, 32, 32, 3)
    # t_2 shape = (4, 32, 32, 3)
    # b_1 shape = (4, 1, 1, 1)
    # b_2 shape = (4, 32, 32, 3)

    return None

def tf_tile():
    """
    broadcasting 内存无关， 而 tile 内存相关,tile会将数据实际复制几份，占用内存。
    用法不同，broadcasting 更简洁。
    注意： tf.tile 的 multiple 参数是扩展倍数
    """
    print("###############3、tf_tile################")
    t = tf.ones([3,4])
    # 直接将t 的tensor扩展成[2,3,4]维度
    t_1 = tf.broadcast_to(t,[2,3,4])
    # 先在 0 位置(起始位置) 添加一个维度,然后使用tile方法将每个维度扩展成之前维度的倍数[2,1,1]
    t_2 = tf.expand_dims(t,axis=0)
    t_2 = tf.tile(t_2,[2,1,1])#此处指的是0维度出扩展2倍，其余维度1倍，保持不变

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    return None

if __name__ == '__main__':
    tf_broadcast()
    tf_broadcast_to()
    tf_tile()

