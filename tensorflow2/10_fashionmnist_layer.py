# -*- coding: utf-8 -*-
# @Time : 2020/1/4 23:39
# @Author : sxp
# @Email : 
# @File : 10_fashionmnist_layer.py
# @Project : python_common

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  datasets,layers,optimizers,Sequential,metrics

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    """数据转换"""
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    return x,y

def check_data(db):
    db_iter = iter(db)
    sample = next(db_iter)
    # 打印一个批次的维度信息 和 批次大小
    print('batch:',sample[0].shape, sample[1].shape)

def get_model():
    """设计模型"""
    model = Sequential([
        layers.Dense(256, activation=tf.nn.relu),  # [b, 784] => [b, 256]
        layers.Dense(128, activation=tf.nn.relu),  # [b, 256] => [b, 128]
        layers.Dense(64, activation=tf.nn.relu),  # [b, 128] => [b, 64]
        layers.Dense(32, activation=tf.nn.relu),  # [b, 64] => [b, 32]
        layers.Dense(10)  # [b, 32] => [b, 10], 330 = 32*10 + 10
    ])
    model.build(input_shape=[None, 28*28])
    return model

def train(db_train, db_test, model, epochs, optimizer):
    """模型训练"""
    for epoch in range(epochs):
        # 训练
        for step,(x, y) in enumerate(db_train):
            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28 * 28])
            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)
                # [b]
                y_onehot = tf.one_hot(y, depth=10)
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss_cross_entropy = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_cross_entropy = tf.reduce_mean(loss_cross_entropy)
            grads = tape.gradient(loss_cross_entropy, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_cross_entropy), float(loss_mse))
        # 验证
        total_correct = 0
        total_num = 0
        for x,y in db_test:
            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            # [b, 10]
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]
        accuracy = total_correct / total_num
        print(epoch, 'test acc:', accuracy)




if __name__ == '__main__':
    """mnist手写字体识别"""
    batch_size = 128
    learning_rate = 1e-3
    epochs = 30

    mnist_path = r'E:\PythonProjects\python_common\tensorflow2\data\mnist.npz'
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(mnist_path)

    print('x train shape = {},y train shape = {}'.format(x_train.shape, y_train.shape))

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
        .map(preprocess).shuffle(10000).batch(batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))\
        .map(preprocess).batch(batch_size)
    # check_data(db_train)

    """模型相关"""
    model =get_model()
    optimizer = optimizers.Adam(lr=learning_rate)
    train(db_train,db_test,model,epochs,optimizer)


