# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_forward.py
@time: 2019/12/28 15:48
"""

import tensorflow as tf
from keras import datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#2只打印错误信息，0打印所有信息

def preprocess(x, y):
    # [b, 28, 28], [b]
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y



def net_forward(epochs,db_train,db_test,lr):
    """前向传播"""
    # 网络的参数
    # 网络结构：[b,28*28] => [b,256] => [b,128] => [b,10]
    # w 和 b 的维度信息
    #w:[dim_in,dim_out]
    # b:[dim_out]
    # 标准正太分布：均值0方差1，但是在这个例子中产生梯度爆炸，可以添加stddev指定方差
    w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
    b1 = tf.Variable(tf.zeros([512]))
    w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
    b2 = tf.Variable(tf.zeros([256]))
    w3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))

    # 前向运算
    for epoch in range(epochs):  # iterate db for 10
        # enumerate()每次计算完一个批次后，step就会+1，即相当于一个批次的编号，所有编号的批次加起来就是整个数据集
        for step, (x, y) in enumerate(db_train):
            # 循环拿到的数据形状 ：x shape = [128,28,28], y shape = [128]
            # h1 = x@w1 + 1 中，要求x的维度为[b,28*28],所以需要进行维度变换
            x = tf.reshape(x,[-1,28*28])
            with tf.GradientTape() as tape:#tensorfl自动求导,但是只会跟中tf.Variable类型的自动求导
                # layer1
                h1 = x@w1 + tf.broadcast_to(b1,[x.shape[0],512])
                h1 = tf.nn.relu(h1)
                # layer2
                h2 = h1@w2 + b2
                h2 = tf.nn.relu(h2)
                # output
                out = h2@w3 + b3
                #计算误差
                # [b, 10] - [b, 10]
                # 求出每个样本的输出和真实值之间的平方和
                loss = tf.square(y - out)
                # [b, 10] => [b]
                loss = tf.reduce_mean(loss, axis=1)
                # [b] => scalar
                loss = tf.reduce_mean(loss)
            # 计算梯度
            # list[]
            grads = tape.gradient(loss,[w1, b1, w2, b2, w3, b3])
            # 梯度更新
            # 原地更新，在w1上直接更新w1
            # update w' = w - lr*grad
            for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
                p.assign_sub(lr * g)
            # 打印输出
            if step % 100 == 0:
                print( 'epoch = {}, step = {},loss = {}'.format(epoch, step, float(loss)))
            """验证集验证"""
            if step % 500 == 0:
                total_correct, total_num = 0, 0
                for step, (x, y) in enumerate(db_test):
                    # layer1.
                    h1 = x @ w1 + b1
                    h1 = tf.nn.relu(h1)
                    # layer2
                    h2 = h1 @ w2 + b2
                    h2 = tf.nn.relu(h2)
                    # output
                    out = h2 @ w3 + b3
                    # [b, 10] => [b]
                    pred = tf.argmax(out, axis=1)
                    # convert one_hot y to number y
                    y = tf.argmax(y, axis=1)
                    # bool type
                    correct = tf.equal(pred, y)
                    # bool tensor => int tensor => numpy
                    total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                    total_num += x.shape[0]
                print(step, 'Evaluate Acc:', total_correct / total_num)


if __name__ == '__main__':
    lr = 1e-3
    epochs = 30
    # x :[60k,28,28]  共有60000张图片，图片大小28*28
    # y:[60k]
    data_dir =os.path.abspath(r'../data/mnist.npz')
    # F:\PythonProjects\python_common\tensorflow2\data\mnist.npz
    print(data_dir)

    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(data_dir)
    print('x_train:', x_train.shape, 'y_train:', y_train.shape, 'x test:', x_test.shape, 'y test:', y_test.shape)
    #转成tensor
    # x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
    # y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)
    # x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
    # y_test = tf.convert_to_tensor(y_test,dtype=tf.int32)
    # 数归一化，x的像素转到 0-1之间
    # x_train = x_train / 255.
    # x_test = x_test / 255.

    # check_dataset_attr(x,y)

    # 构建训练数据
    db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(60000).batch(128).map(preprocess).repeat(epochs)
    db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(10000).batch(128).map(preprocess)
    x,y = next(iter(db_train))
    print('train sample:', x.shape, y.shape)


    print('train data count = {}'.format(len(x_train)))
    print('test data count = {}'.format(len(x_test)))
    #前向传播算法
    net_forward(epochs,db_train,db_test,lr)









