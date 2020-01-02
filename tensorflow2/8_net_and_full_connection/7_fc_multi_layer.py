# encoding: utf-8

"""
@author: sunxianpeng
@file: 7_fc_multi_layer.py
@time: 2020/1/2 14:06
"""

import tensorflow as tf
from keras.layers import Dense
from keras import Sequential

def tf_sequential():
    """ 　
    Sequential类的属性：
　　　　layers: (功能待查)
　　　　run_eagerly: 可设置属性，指示模型是否应该急切运行
　　　　sample_weights： 设置样本权重
　　　　state_updates： 更新所有有状态的图层并返回。这对于分离训练更新和状态更新很有用，例如，当我们需要在预测期间更新图层的内部状态时。
　　　　stateful：(功能待查)
　　Sequential类的方法：
　　　　add(): 添加图层，参数layer：图层实例, 对比类属性layers我们应该知道属性在实例化的时候是可以直接传入一个网络结构；
　　　　compile()：编译模型"""
    print("########################1、tf_sequential #######################")
    x = tf.random.normal([2, 512])
    # 三层网络，
    model = Sequential([
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(10)
    ])
    # 设置输入维度 4
    model.build(input_shape=[None, 512])
    # 查看网络结构
    model.summary()

    out = model(x)

    print('out shape = {}'.format(out.shape))
    return None


if __name__ == '__main__':
    tf_sequential()