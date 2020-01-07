# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_keras_metrics.py
@time: 2020/1/6 14:34
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
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
    """模型训练,测试评价标准的使用"""
    for epoch in range(epochs):
        """建立测量尺"""
        loss_meter = metrics.Mean()
        accuracy_meter = metrics.Accuracy()

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
                loss_regularization = []
                for p in model.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))
                loss_cross_entropy = loss_cross_entropy + 0.0001 * loss_regularization
                """更新数据"""
                loss_meter.update_state(loss_cross_entropy)

            grads = tape.gradient(loss_cross_entropy, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                """获取数据并清除buffer"""
                print('epoch = {},step = {},cross loss = {},mse loss = {}'
                      .format(epoch,step,loss_meter.result().numpy(),float(loss_mse)))
                loss_meter.reset_states()

        # 准确率测量尺和下面自定义的 total_correct，total_num功能相同
        total_correct = 0
        total_num = 0
        """清除buffer"""
        accuracy_meter.reset_states()
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
            accuracy_meter.update_state(y,pred)
        accuracy = total_correct / total_num
        """获取数据"""
        print('##########epoch = {},evaluate accuracy meter = {}, test acc = {}'
              .format(epoch, accuracy_meter.result().numpy(), accuracy))

if __name__ == '__main__':
    """使用测量尺"""
    batch_size = 128
    learning_rate = 1e-3
    epochs = 1

    mnist_path =os.path.abspath(r'../data/mnist.npz')
    print('mnist path = {}'.format(mnist_path))

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