# encoding: utf-8

"""
@author: sunxianpeng
@file: 4_preceptron.py
@time: 2020/1/4 19:09
"""
import tensorflow as tf

def tf_single_output_perceptron():
    """单层感知机梯度"""
    print("###################1、tf_single_output_perceptron ##################")
    x = tf.random.normal([1, 3])
    w = tf.ones([3, 1])
    b = tf.ones([1])
    y = tf.constant([1])
    with tf.GradientTape() as tape:
        tape.watch([w, b])
        z = x @ w + b
        logits = tf.sigmoid(z)
        loss = tf.reduce_mean(tf.losses.MSE(y, logits))
    grads = tape.gradient(loss, [w, b])

    print('w grad = {}'.format(grads[0]))
    print("================")
    print('b grad = {}'.format(grads[1]))
    # w grad = [[ 0.3951622 ]
    #  [ 0.20868859]
    #  [-0.11809778]]
    # ================
    # b grad = [-0.29602545]
    return None

def tf_multi_output_perceptron():
    """多层感知机梯度 """
    print("###################2、tf_multi_output_perceptron ##################")
    x = tf.random.normal([2, 4])
    w = tf.ones([4, 3])
    b = tf.ones([3])
    y = tf.constant([2, 0])

    with tf.GradientTape() as tape:
        tape.watch([w, b])
        prob = tf.nn.softmax(x @ w + b)
        loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y,depth=3), prob))
    grads = tape.gradient(loss, [w, b])

    print('w grad = {}'.format(grads[0]))
    print("================")
    print('b grad = {}'.format(grads[1]))
    # w grad = [[ 0.1718041  -0.03977256 -0.13203155]
    #  [-0.01751113  0.00112793  0.01638321]
    #  [-0.02479583  0.00388955  0.02090628]
    #  [ 0.04892458 -0.03387642 -0.01504816]]
    # ================
    # b grad = [-0.03703704  0.07407407 -0.03703704]
    return None

if __name__ == '__main__':
    tf_single_output_perceptron()
    tf_multi_output_perceptron()