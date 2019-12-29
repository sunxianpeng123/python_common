# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_equal.py
@time: 2019/12/29 21:08
"""
import numpy as np
import tensorflow as tf

def tf_equal():
    """ 实现功能：比较两个tensor，并统计出相同元素的个数"""
    print("####################1、tf_equal ###########################")
    a = tf.constant([1,2,3,2,5])

    arr_1 = np.asarray([1,2,2,2,5])
    b = tf.convert_to_tensor(arr_1)

    #
    c = tf.equal(a,b)
    # 将 True 和 False 转为 0,1整数
    d = tf.cast(c,dtype=tf.int32)
    e = tf.reduce_sum(d)

    print('a = {}'.format(a))
    print('b = {}'.format(b))
    print('c = {}'.format(c))
    print('d = {}'.format(d))
    print('e = {}'.format(e))
    # a = [1 2 3 2 5]
    # b = [1 2 2 2 5]
    # c = [ True  True False  True  True]
    # d = [1 1 0 1 1]
    # e = 4
    return None

def tf_accuracy():
    """ 求解准确率"""
    print("####################2、tf_accuracy ###########################")
    a = tf.constant([[0.1,0.2,0.7],
                     [0.9,0.05,0.05]])#shape = (2,3)
    print(a)
    # tf.Tensor(
    # [[0.1  0.2  0.7 ]
    #  [0.9  0.05 0.05]], shape=(2, 3), dtype=float32)

    # 求出每行最大值的索引位置，相当于取出每个样本最大概率的分类，转成int32
    pred = tf.cast(tf.argmax(a,axis=1),dtype=tf.int32)
    # 已知样本的分类标签
    y = tf.constant([2,1])
    # 预测结果和真实结果进行比较，
    res = tf.equal(y,pred)
    # 计算出预测正确的样本个数
    correct = tf.reduce_sum(tf.cast(res,dtype=tf.int32))
    # 求解准确度，预测对的个数除以总的个数
    accuracy = correct / 2
    print('accuracy = {}'.format(accuracy))

    print(correct)
    return None

def tf_unique():
    """去除tensor中的重复元素"""
    print("####################3、tf_unique ###########################")
    a = tf.range(5)
    b = tf.unique(a)

    print('a = {}'.format(a))
    # 返回的数据中包含
    print('b = {}'.format(b))
    # b = Unique(y= < tf.Tensor: id = 31, shape = (5,), dtype = int32, numpy = array(
    #     [0, 1, 2, 3, 4]) >, idx = < tf.Tensor: id = 32, shape = (5,), dtype = int32, numpy = array([0, 1, 2, 3, 4]) >)
    print("=====================")
    c = tf.constant([4,2,2,4,3])
    d = tf.unique(c)
    # 将 d 还原回 c,
    # 传入去重后的数据，和 去重前的数据在去重后的数据中的下标索引
    e = tf.gather(d.y,d.idx)


    print('c = {}'.format(c))
    print('d = {}'.format(d))
    print('d.idx = {}'.format(d.idx))#去重前的数据在去重后的数据中的位置
    print('d.y = {}'.format(d.y))#去重后的tensor
    print('e = {}'.format(e))
    # c = [4 2 2 4 3]

    # d = Unique(y= < tf.Tensor: id = 37, shape = (3,), dtype = int32, numpy = array(
    #     [4, 2, 3]) >, idx = < tf.Tensor: id = 38, shape = (5,), dtype = int32, numpy = array([0, 1, 1, 0, 2]) >)

    # d.idx = [0 1 1 0 2]
    # d.y = [4 2 3]
    # e = [4 2 2 4 3]



    return None


if __name__ == '__main__':
    tf_equal()
    tf_accuracy()
    tf_unique()