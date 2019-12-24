# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_selective_gather_and_nd.py
@time: 2019/12/24 19:58
"""
import tensorflow as tf
# 有选择的索引数据
def tf_boolean_mask():
    """ 指定某个维度和在该维度上选择的索引号，获取数据，
    即指定班级或者学生或者课程的id获取数据"""
    print("##############1、tf_gather################")
    t = tf.random.normal([4,28,28,3])#4张28*28*3的图片，RGB通道
    # 取第一张和第三张的图片数据
    t_1 = tf.boolean_mask(t,mask=[True,False,True,False])
    #取所有图片的 第四个维度的R通道和G通道的数据
    t_2 = tf.boolean_mask(t,mask=[True,True,False],axis=3)


    a = tf.ones([2,3,4])
    # mask组成的矩阵对应a中的前两个维度
    # 取a中第1维度上的第2维度第一个位置对应的第三个维度所有数据（4个数据）
    # 和 第1维度上第2维度第二和第三个位置对应的第三个维度上的所有数据(2*4个数据)
    # 共取了三个样本，并且每个样本四个数据，所以是3*4 的结果
    a_1 = tf.boolean_mask(a,mask=[[True,False,False],[False,True,True]])
    print(a_1)

    print('t shape = {}'.format(t.shape))
    print('t_1 shape = {}'.format(t_1.shape))
    print('t_2 shape = {}'.format(t_2.shape))
    print('a_1 shape = {}'.format(a_1.shape))
    return None


if __name__ == '__main__':
    tf_boolean_mask()
