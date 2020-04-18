# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_normal():
    """正态分布随机初始化"""
    print("##############1、正态分布###################")
    #
    # shape: 一维的张量，也是输出的张量。
    # mean: 正态分布的均值。
    # stddev: 正态分布的标准差。
    # dtype: 输出的类型。
    # seed: 一个整数，当设置之后，每次生成的随机数都一样。
    # name: 操作的名字。

    # 均值和方差均为 1
    t_1 = tf.random.normal([2,2],mean=1,stddev=1)
    # 均值为0 方差为1
    t_2 = tf.random.normal([2,2])
    # 从截断的正态分布中输出随机值。
    # 生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。
    #
    # 在正态分布的曲线中，横轴区间（μ-σ，μ+σ）内的面积为68.268949%。
    # 横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。
    # 横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。
    # X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。
    # 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
    t_3 = tf.random.truncated_normal([2,2],mean=0,stddev=1)

    print(t_1.dtype)
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_1 = {}'.format(t_1))
    print(t_2.dtype)
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_2 = {}'.format(t_2))
    print(t_3.dtype)
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_3 = {}'.format(t_3))
    # <dtype: 'float32'>
    # t_1 shape = (2, 2)
    # t_1 = [[ 0.0643813  1.7637568]
    #  [ 0.8251718 -0.3416227]]
    # <dtype: 'float32'>
    # t_2 shape = (2, 2)
    # t_2 = [[ 1.1587772  -0.04811676]
    #  [ 0.33243996  0.14522554]]
    # <dtype: 'float32'>
    # t_3 shape = (2, 2)
    # t_3 = [[-0.15677835  0.27606738]
    #  [-1.629522   -0.39258146]]

    return None

def tf_uniform():
    """均匀分布"""
    print("##############2、均匀分布###################")

    t_1 = tf.random.uniform([2,2],minval=0,maxval=1)
    t_2 = tf.random.uniform([2,2],minval=0,maxval=100,dtype=tf.int32)
    print(t_1)
    print(t_1.dtype)
    print('t_1 shape = {}'.format(t_1.shape))
    print(t_2)
    print(t_2.dtype)
    print('t_2 shape = {}'.format(t_2.shape))
    # <dtype: 'float32'>
    # t_1 shape = (2, 2)
    # <dtype: 'float32'>
    # t_2 shape = (2, 2)
    return None

def random_permutation():
    """随机打散"""
    print("##############3、随机打散###################")
    idx = tf.range(10)#10张照片
    print('before shuffle = {}'.format(idx))
    idx = tf.random.shuffle(idx)
    print('after shuffle = {}'.format(idx))
    # 照片的特征 784 维度
    data = tf.random.normal([10,784])
    print(data.shape)
    # 从原始数据中取出打散后的前五条记录
    train_1 = tf.gather(data,idx[:5])
    print(train_1.shape)



    return None

if __name__ == '__main__':
    tf_normal()
    tf_uniform()
    random_permutation()