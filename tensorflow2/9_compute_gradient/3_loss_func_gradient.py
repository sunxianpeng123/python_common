# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_loss_func_gradient.py
@time: 2020/1/4 16:55
"""
import tensorflow as tf
def tf_mse_gradient():
    """ 均方差损失函数梯度"""
    print("################1、tf_mse_gradient###################")
    x = tf.random.normal([2, 4])
    w = tf.random.normal([4, 3])
    b = tf.zeros([3])
    # 两个样本对应的标签
    y = tf.constant([2, 0])

    with tf.GradientTape() as tape:
        tape.watch([w, b])
        logits = x @ w + b
        prob = tf.nn.softmax(logits, axis=1)
        loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth=3), prob))
    grads = tape.gradient(loss, [w, b])

    print('grads type = {}'.format(type(grads)))
    print(grads[0])
    print("=================")
    print(grads[1])
    # grads type = <class 'list'>

    # tf.Tensor(
    # [[ 0.04673104  0.01906405 -0.06579509]
    #  [-0.08409811  0.02800278  0.05609531]
    #  [-0.00172295  0.00875067 -0.00702773]
    #  [ 0.01617242 -0.01706368  0.00089126]], shape=(4, 3), dtype=float32)
    # =================
    # tf.Tensor([ 0.00084885  0.04564979 -0.04649865], shape=(3,), dtype=float32)
    return None

def tf_cross_entropy_gradient():
    """交叉熵损失函数梯度"""
    print("################2、tf_cross_entropy_gradient ###################")
    x = tf.random.normal([2, 4])
    w = tf.random.normal([4, 3])
    b = tf.zeros([3])
    # 两个样本对应的标签
    y = tf.constant([2, 0])

    with tf.GradientTape() as tape:
        tape.watch([w, b])
        logits = x @ w + b
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(tf.one_hot(y,depth=3), logits, from_logits=True))
    grads = tape.gradient(loss, [w, b])

    print('grads type = {}'.format(type(grads)))
    print("=================")
    print(grads[0])
    print("=================")
    print(grads[1])
    # grads type = <class 'list'>
    # =================
    # tf.Tensor(
    # [[ 0.58834654 -0.17131177 -0.4170348 ]
    #  [-0.09703365 -0.01219718  0.10923082]
    #  [ 0.25620204 -0.0910174  -0.16518466]
    #  [-0.28719124 -0.12367874  0.41086996]], shape=(4, 3), dtype=float32)
    # =================
    # tf.Tensor([-0.284351    0.23283815  0.05151287], shape=(3,), dtype=float32)
    return None

if __name__ == '__main__':
    tf_mse_gradient()
    tf_cross_entropy_gradient()