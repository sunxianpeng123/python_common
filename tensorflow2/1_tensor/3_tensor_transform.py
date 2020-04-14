# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_tensor_transform.py
@time: 2019/12/22 20:26
"""
import tensorflow as tf
import numpy as np

def np_tensor_transform():
    """numpy和tensor之间转换"""
    print("##############1、numpy和tensor之间转换#################")
    a = np.arange(5)
    a_t_1 = tf.convert_to_tensor(a)
    a_t_2 = tf.convert_to_tensor(a,dtype=tf.int32)

    print('a = {}'.format(a))
    print('a.dtype = {}'.format(a.dtype))
    print('a_t_1 = {}'.format(a_t_1))
    print('a_t_1.dtype = {}'.format(a_t_1.dtype))
    print('a_t_2 = {}'.format(a_t_2))
    print('a_t_2.dtype = {}'.format(a_t_2.dtype))

    # a = [0 1 2 3 4]
    # a.dtype = int32
    # a_t_1 = [0 1 2 3 4]
    # a_t_1.dtype = <dtype: 'int32'>
    # a_t_2 = [0 1 2 3 4]
    # a_t_3.dtype = <dtype: 'int32'>
    print("##############2、tensor 内转换#################")
    a_t_3 = tf.cast(a_t_1,dtype=tf.float32)
    print('a_t_3 = {}'.format(a_t_3))
    print('a_t_3.dtype = {}'.format(a_t_3.dtype))
    # a_t_3 = [0. 1. 2. 3. 4.]
    # a_t_3.dtype = <dtype: 'float32'>
    return None

def bool_int_transofrm():
    """布尔和int类型之间的转换"""
    print("####################3、布尔和int类型之间的转换####################")
    # 0 --> false, 1--->True
    a = tf.constant([0,1])
    a_bool_1 = tf.cast(a,dtype=tf.bool)

    print('a = {}'.format(a))
    print('a_bool_1 = {}'.format(a_bool_1))

    # a = [0 1]
    # a_bool_1 = [False  True]
    return None



if __name__ == '__main__':
    np_tensor_transform()
    bool_int_transofrm()

