# encoding: utf-8

"""
@author: sunxianpeng
@file: 4_variable.py
@time: 2019/12/22 20:51
"""

import tensorflow as tf
import numpy as np

"""
本身是一个tensor，再次进行了一次包装，可以进行求导
"""

def variable_info():
    """variable相关操作"""
    print("################1、variable相关操作################")
    a = tf.range(5)
    b = tf.Variable(a)
    c = tf.Variable(a,name='input_data')

    print('a = {}'.format(a))
    print('a.dtype = {}'.format(a.dtype))
    print('b = {}'.format(b))
    print('b.dtype = {}'.format(b.dtype))
    print('c = {}'.format(c))
    print('c.name = {}'.format(c.name))
    print('c.trainable = {}'.format(c.trainable))#是否需要求导，包含梯度信息
    # a = [0 1 2 3 4]
    # a.dtype = <dtype: 'int32'>
    # b = <tf.Variable 'Variable:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])>
    # b.dtype = <dtype: 'int32'>
    # c = <tf.Variable 'input_data:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])>
    # c.name = input_data:0
    # c.trainable = True
    print("################2、variable类型检查################")
    print(isinstance(b,tf.Tensor))#不推荐使用此方法判断类型
    print(isinstance(b,tf.Variable))
    print(tf.is_tensor(b))
    # False
    # True
    # True
    return None

def np_variable_transform():
    print("################3、numpy转variable################")
    a = tf.range(5)
    b = tf.Variable(a)

    print('a = {}'.format(a))
    print('a.dtype = {}'.format(a.dtype))
    print('b = {}'.format(b))
    print('b.dtype = {}'.format(b.dtype))

    # a = [0 1 2 3 4]
    # a.dtype = <dtype: 'int32'>
    # b = <tf.Variable 'Variable:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])>
    # b.dtype = <dtype: 'int32'>
    print("################4、variable转numpy################")
    a_np = a.numpy()
    b_np = b.numpy()
    print('a_np = {}'.format(a_np))
    print('a_np.dtype = {}'.format(a_np.dtype))
    print('b_np = {}'.format(b_np))
    print('b_np.dtype = {}'.format(b_np.dtype))

    # a_np = [0 1 2 3 4]
    # a_np.dtype = int32
    # b_np = [0 1 2 3 4]
    # b_np.dtype = int32
    print("################5、tensor标量类型转换################")
    c = tf.ones([])
    c_int = int(c)
    c_float = float(c)

    print(c)
    print('c = {}'.format(c))
    print('c.dtype = {}'.format(c.dtype))
    print(c_int)
    print('c_int = {}'.format(c_int))
    print('c_int type = {}'.format(type(c_int)))
    print(c_float)
    print('c_float = {}'.format(c_float))
    print('c_float type = {}'.format(type(c_float)))

    # c = 1.0
    # c.dtype = <dtype: 'float32'>
    # c_int = 1
    # c_int type = <class 'int'>
    # c_float = 1.0
    # c_float type = <class 'float'>
    return None


if __name__ == '__main__':
    variable_info()
    np_variable_transform()
