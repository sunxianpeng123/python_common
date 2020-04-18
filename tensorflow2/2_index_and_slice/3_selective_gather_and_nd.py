# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_selective_gather_and_nd.py
@time: 2019/12/24 19:58
"""
import tensorflow as tf
# 有选择的索引数据
def tf_gather():
    """ 指定某个维度和在该维度上选择的索引号，获取数据，
    即指定班级或者学生或者课程的id获取数据"""
    print("##############1、tf_gather################")
    t = tf.random.normal([4,35,8])#4个班级，每个班级35个学生，八门课程
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

def tf_gahter_nd():
    """指定多个维度和在多个维度上的索引号的对应关系，获取数据，
    即比如指定班级和学生索引的对应关系，则可以实现获取不同班级对应不同学生的课程"""
    print("##############1、tf_gahter_nd################")
    t = tf.random.normal([4,35,8])#4个班级，每个班级35个学生，八门课程
    # 取 0 号班级的所有学生的所有课程
    t_1 = tf.gather_nd(t,[0])
    # 取0 班级的 1号学生的所有课程
    t_2 = tf.gather_nd(t,[0,1])
    # 取0 班级的 1号学生,2号课程
    t_3 = tf.gather_nd(t,[0,1,2])
    # 取 0 班级的 1号学生,2号课程 和 1班级的 2号学生,3号课程
    t_4 = tf.gather_nd(t,[[0,1,2],[1,2,3]])
    print(t_4)#tf.Tensor([ 0.5307846  -0.40341568], shape=(2,), dtype=float32)

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('t_3 shape = {}'.format(t_3.shape))
    print('t_4 shape = {}'.format(t_4.shape))

    return  None

if __name__ == '__main__':
    tf_gather()
    tf_gahter_nd()
