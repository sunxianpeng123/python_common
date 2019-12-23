# encoding: utf-8

"""
@author: sunxianpeng
@file: 6_create_tensor_zeros.py
@time: 2019/12/23 19:46
"""
import tensorflow as tf
import numpy as np

def tf_scalar():
    """标量"""
    print("##############1、tf_scalar###################")
    out = tf.random.uniform([4,10])#神经网络输出，4张照片，10个分类的得分
    y = tf.range(4)#4张照片的标签分类
    # 独热编码，即哪个分类有值，哪个位置就是1，其余是0，
    # depth=10表示扩充成10维度
    y = tf.one_hot(y,depth=10)
    # 求均方差
    loss = tf.keras.losses.mse(y,out)
    # 输出每个图片对应的均方差
    print(loss)#tf.Tensor([0.28632194 0.3595503  0.462607   0.31716153], shape=(4,), dtype=float32)
    # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
    # 第一个参数input_tensor： 输入的待降维的tensor;
    # 第二个参数axis： 指定的轴，如果不指定，则计算所有元素的均值;
    # 第三个参数keep_dims：是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
    # 第四个参数name： 操作的名称;
    # 第五个参数 reduction_indices：在以前版本中用来指定轴，已弃用;
    # 类似函数还有:
    #
    #     tf.reduce_sum ：计算tensor指定轴方向上的所有元素的累加和;
    #     tf.reduce_max  :  计算tensor指定轴方向上的各个元素的最大值;
    #     tf.reduce_all :  计算tensor指定轴方向上的各个元素的逻辑和（and运算）;
    #     tf.reduce_any:  计算tensor指定轴方向上的各个元素的逻辑或（or运算）;

    loss = tf.reduce_mean(loss)#tf.Tensor(0.325341, shape=(), dtype=float32)
    print(loss)

    return None


if __name__ == '__main__':
    tf_scalar()