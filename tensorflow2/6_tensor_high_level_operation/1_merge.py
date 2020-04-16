# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_merge.py
@time: 2019/12/28 19:31
"""


import tensorflow as tf

def tf_concat():
    """拼接操作"""
    print("################1、tf_concat ##################")
    a = tf.ones([4, 35, 8])#4个班级，35个学术，8门课程
    b = tf.ones([2, 35, 8])
    c = tf.ones([1, 35, 8])
    d = tf.ones([4, 32, 8])
    e = tf.ones([4, 3, 8])

    # 将多个数据集合并成一个,axis代表维度下标
    # axis=0 表示以第一个维度为标准，其余的维度必须相同
    # axis=1 表示以第二个维度为标准，其余的维度必须相同
    t_1 = tf.concat([a,b,c],axis=0)
    t_2 = tf.concat([d,e],axis=1)

    print('t_1 shape = {}'.format(t_1.shape))#t_1 shape = (7, 35, 8)
    print('t_2 shape = {}'.format(t_2.shape))#t_2 shape = (4, 35, 8)

    return None

def tf_stack():
    """堆叠操作，若合并数据集时，需要  添加新的维度  适用"""
    print("################2、tf_stack ##################")
    a = tf.ones([4, 35, 8])#school 1 学校的信息：4个班级，35个学术，8门课程
    b = tf.ones([4, 35, 8])#school 2 学校的信息：4个班级，35个学术，8门课程
    # c = tf.ones([1, 35, 8])
    # d = tf.ones([4, 32, 8])
    # e = tf.ones([4, 3, 8])

    """需要添加一个学校的维度"""
    # axis=0表示在第一个维度4之前添加一个学校维度
    # [4, 35, 8]共有四个位置可以添加新的维度，四个位置即：开头、结尾和两个空隙
    # 两个tensor已经存在的维度必须全部相同，才可以使用stack方法
    t_1 = tf.stack([a,b],axis=0)
    t_2 = tf.stack([a,b],axis=3)#t_1 shape = (2, 4, 35, 8)


    print('t_1 shape = {}'.format(t_1.shape))# t_1 shape = (2, 4, 35, 8)
    print('t_2 shape = {}'.format(t_2.shape))# t_2 shape = (4, 35, 8, 2)


    return None

if __name__ == '__main__':
    tf_concat()
    tf_stack()