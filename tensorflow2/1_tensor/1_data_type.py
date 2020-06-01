# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_data_type.py
@time: 2019/12/22 19:42
"""

import tensorflow as tf
import numpy as np
"""
标量:零维数据
    scalar:1.1
向量
    vector：[1.1],[1.1,2.2,…]
多维向量
    matrix:[[1,1],[2,2],[3,3],…]
    在tensorflow中，只要是向量，都算tensor
支持数据类型
        int,float,double
        bool
        string
"""

def constant():
    """常量"""
    print('#################1、创建常量####################')
    print("数值常量===================================")
    constant_1 = tf.constant(1)
    print(constant_1)
    print(constant_1.shape == ())
    print('constant_1 = {}'.format(constant_1))

    constant_2 = tf.constant(1.)
    print(constant_2)
    print('constant_2 = {}'.format(constant_2))

    # constant_3 = tf.constant(2.2,dtype=tf.int32) #TypeError: Cannot convert 2.2 to EagerTensor of dtype int32

    constant_4 = tf.constant(2.,dtype=tf.double)
    print(constant_4)
    print('constant_4 = {}'.format(constant_4))
    # tf.Tensor(1, shape=(), dtype=int32)
    # constant_1 = 1
    # tf.Tensor(1.0, shape=(), dtype=float32)
    # constant_2 = 1.0
    # tf.Tensor(2.0, shape=(), dtype=float64)
    # constant_4 = 2.0
    print("布尔常量===================================")
    bool_1 = tf.constant([True,False])
    print(bool_1.shape == (2,))
    print(bool_1)
    print('bool_1 = {}'.format(bool_1))
    # tf.Tensor([ True False], shape=(2,), dtype=bool)
    # bool_1 = [ True False]
    print("字符串常量===================================")
    str_1 = tf.constant('hello,world.')
    print(str_1)
    print('str_1 = {}'.format(str_1))
    # tf.Tensor(b'hello,world.', shape=(), dtype=string)
    # str_1 = b'hello,world.'

    return None

def check_data_type():
    """检查数据类型"""
    print("###################2、检查数据类型#######################")
    a = tf.constant([1.])
    b = tf.constant([True,False])
    c = tf.constant('hello,world.')
    d = np.arange(4)
    print("是否是张量=================================================")
    print('a instance = {}'.format(isinstance(a,tf.Tensor)))
    print('b instance = {}'.format(tf.is_tensor(b)))
    print('d instance = {}'.format(tf.is_tensor(d)))

    # a instance = True
    # b instance = True
    # d instance = False
    print("数据类型比较=================================================")
    print('a.dtype = {}'.format(a.dtype))
    print('b.dtype = {}'.format(b.dtype))
    print('c.dtype = {}'.format(c.dtype))
    print(a.dtype == tf.float32)
    print(c.dtype == tf.string)
    # a.dtype = <dtype: 'float32'>
    # b.dtype = <dtype: 'bool'>
    # c.dtype = <dtype: 'string'>
    # True
    # True
    return None


if __name__ == '__main__':
    constant()
    check_data_type()

