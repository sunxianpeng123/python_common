# encoding: utf-8

"""
@author: sunxianpeng
@file: 3_vgg13.py
@time: 2020/1/10 11:36
"""

import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

def prepross(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y
 
def get_vgg13_model():
    """vgg13, 13指的是卷积层和全连接层"""
    """卷积层"""
    conv_net = [  # 5 units of conv + max pooling
        # unit 1
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 2
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 3
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 4
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        # unit 5
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
    ]
    """全连接层"""
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(100, activation=None),
    ])
    """  """
    conv_net = Sequential(conv_net)
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    return conv_net, fc_net

def train(db_train, db_test, conv_net, fc_net, epochs, optimizer):
    """模型训练,测试评价标准的使用"""
    # [1, 2] + [3, 4] => [1, 2, 3, 4]
    variables = conv_net.trainable_variables + fc_net.trainable_variables
    for epoch in range(epochs):
        """建立测量尺"""
        optimizer.learning_rate = 0.2 * (100 - epoch) / 100
        print('optimizer.learning_rate = {}'.format(optimizer.learning_rate))

        for step,(x, y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                # [b, 28, 38, 3] => [b, 1, 1, 512]
                out = conv_net(x)
                # flatten, => [b, 512]
                out = tf.reshape(out, [-1, 512])
                # [b, 512] => [b, 10]
                logits = fc_net(out)
                # [b] => [b, 10]
                y_onehot = tf.one_hot(y, depth=100)
                loss_cross_entropy = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_cross_entropy = tf.reduce_mean(loss_cross_entropy)
                loss_regularization = []
                for p in variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
                loss_cross_entropy = loss_cross_entropy + 0.0001 * loss_regularization

            grads = tape.gradient(loss_cross_entropy,variables)
            optimizer.apply_gradients(zip(grads,variables))
            if step % 100 == 0:
                """获取数据并清除buffer"""
                print('epoch = {},step = {},loss_cross_entropy = {}'.format(epoch,step,float(loss_cross_entropy)))

        # 准确率测量尺和下面自定义的 total_correct，total_num功能相同
        total_num = 0
        total_correct = 0
        for x,y in db_test:

            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        accuracy = total_correct / total_num
        print(epoch, 'accuracy:', accuracy)
        """获取数据"""
        print('##########epoch = {},test acc = {}'.format(epoch, accuracy))

if __name__ == '__main__':
    #######################################################
    # 参数配置
    #######################################################
    epochs = 15
    batch_size = 128
    learning_rate = 1e-3
    onehot_depth = 10
    save_weights_path = 'ckpt/weights.ckpt'
    mnist_path = os.path.abspath(r'../data/mnist.npz')
    print('mnist path = {}'.format(mnist_path))
    #######################################################
    # 数据准备
    #######################################################
    # [50k, 32, 32, 3], [10k, 1]
    # (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(mnist_path)
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    y_train = tf.squeeze(y_train)
    y_test = tf.squeeze(y_test)
    y_train = tf.one_hot(y_train, depth=onehot_depth)  # [50k, 10]
    y_test = tf.one_hot(y_test, depth=onehot_depth)  # [10k, 10]
    print('datasets:', x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train.min(), x_train.max())

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(prepross).shuffle(10000).batch(batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(prepross).batch(batch_size)
    sample = next(iter(db_train))
    print('batch:', sample[0].shape, sample[1].shape)

    conv_net, fc_net = get_vgg13_model()
    optimizer = optimizers.RMSprop(lr=learning_rate, momentum=0.9)
    train(db_train,db_test,conv_net,fc_net,epochs,optimizer)
