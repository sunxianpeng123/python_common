# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_selective_gather.py
@time: 2019/12/24 19:58
"""
import tensorflow as tf

def tf_gather():
    """ """
    print("##############1、tf_gather################")
    t = tf.random.normal([4,35,8])
    t_1 = t[2:4]
    # axis指定维度，0 代表第一维度，即4 所在维度，按照indices指定的顺序采集数据
    t_2 = tf.gather(t,axis=0,indices=[2,1,3,0])
    t_3 = tf.gather(t,axis=1,indices=[2,3,7,9,16])
    t_4 = tf.gather(t,axis=2,indices=[2,3,7])

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_4 shape = {}'.format(t_4.shape))
    return None


# 有选择的索引数据
if __name__ == '__main__':
    tf_gather()
