# encoding: utf-8

"""
@author: sunxianpeng
@file: 2_index_special.py
@time: 2019/12/24 19:20
"""
import tensorflow as tf

def tf_negtive_index():
    """step 为负数"""
    # ::-1  <=> a:b:-1,从a到b倒序
    print("#############1、step 为负数#################")
    t = tf.range(4)
    t_1 = t[::-1]#step 为 -1，表示从后向前取数据
    t_2 = t[::-2]#step 为 -2，表示从后向前，每隔一步取一个数据
    # [0 1 2 3]
    t_3 = t[2::-2]#从下标索引为2的位置处，即2处，倒序每隔一步取一个数据

    print(t)#tf.Tensor([0 1 2 3], shape=(4,), dtype=int32)
    print(t_1)
    print(t_2)
    print(t_3)
    # tf.Tensor([0 1 2 3], shape=(4,), dtype=int32)
    # tf.Tensor([3 2 1 0], shape=(4,), dtype=int32)
    # tf.Tensor([3 1], shape=(2,), dtype=int32)
    # tf.Tensor([2 0], shape=(2,), dtype=int32)
    return  None

def tf_three_point():
    """三个点的作用"""
    print("#############2、三个点的作用#################")
    t = tf.random.normal([2,4,25,28,3])# 2个学校4个年级25个班级28个学生3们课程的成绩
    t_1 = t[0,:,:,:,:]
    t_2 = t[0,...]

    t_3 = t[:,:,:,:,0]
    t_4 = t[...,0]

    t_5 = t[0,...,2]
    t_6 = t[1,0,...,0]

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_4 shape = {}'.format(t_4.shape))
    print('t_5 shape = {}'.format(t_5.shape))
    print('t_6 shape = {}'.format(t_6.shape))
    # t shape = (2, 4, 25, 28, 3)
    # t_1 shape = (4, 25, 28, 3)
    # t_2 shape = (4, 25, 28, 3)
    # t_3 shape = (2, 4, 25, 28)
    # t_4 shape = (2, 4, 25, 28)
    # t_5 shape = (4, 25, 28)
    # t_6 shape = (25, 28)
    return None


if __name__ == '__main__':
    tf_negtive_index()
    tf_three_point()
