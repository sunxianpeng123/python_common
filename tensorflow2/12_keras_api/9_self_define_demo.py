# encoding: utf-8

"""
@author: sunxianpeng
@file: 9_self_define_demo.py
@time: 2020/1/7 15:25
"""
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def prepross(x, y):
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

class MyDense(layers.Layer):
    # to replace standard layers.Dense()
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        # 去掉了偏执量
        # self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):
        x = inputs @ self.kernel
        return x

class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(img_size*img_size*channel_size, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self, inputs, training=None):
        """
        :param inputs: [b, 32, 32, 3]
        :param training:
        :return:
        """
        x = tf.reshape(inputs, [-1, img_size*img_size*channel_size])
        # [b, 32*32*3] => [b, 256]
        x = self.fc1(x)
        x = tf.nn.relu(x)
        # [b, 256] => [b, 128]
        x = self.fc2(x)
        x = tf.nn.relu(x)
        # [b, 128] => [b, 64]
        x = self.fc3(x)
        x = tf.nn.relu(x)
        # [b, 64] => [b, 32]
        x = self.fc4(x)
        x = tf.nn.relu(x)
        # [b, 32] => [b, 10]
        x = self.fc5(x)
        return x


def train_and_save_weights():
    """模型"""
    model = MyNetwork()

    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    model.fit(db_train, epochs=epochs,validation_data=db_test, validation_freq=1)

    """评估保存"""
    model.evaluate(db_test)
    model.save_weights(save_weights_path)
    del model
    print(save_weights_path)



if __name__ == '__main__':
    is_need_train = False
    #######################################################
    # 参数配置
    #######################################################
    epochs=15
    batch_size = 128
    learning_rate = 1e-3
    onehot_depth = 10
    img_size = 28
    channel_size = 1
    save_weights_path = 'ckpt/weights.ckpt'
    mnist_path =os.path.abspath(r'../data/mnist.npz')
    print('mnist path = {}'.format(mnist_path))
    #######################################################
    # 数据准备
    #######################################################
    # [50k, 32, 32, 3], [10k, 1]
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(mnist_path)
    y_train = tf.squeeze(y_train)
    y_test = tf.squeeze(y_test)
    y_train = tf.one_hot(y_train,depth=onehot_depth)# [50k, 10]
    y_test = tf.one_hot(y_test,depth=onehot_depth)# [10k, 10]
    print('datasets:', x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_train.min(), x_train.max())

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(prepross).shuffle(10000).batch(batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(prepross).batch(batch_size)
    sample = next(iter(db_train))
    print('batch:', sample[0].shape, sample[1].shape)
    #######################################################
    # 模型优化器，损失函数等
    #######################################################
    optimizer = optimizers.Adam(lr=learning_rate)
    loss = tf.losses.CategoricalCrossentropy(from_logits=True)
    #######################################################
    # 是否训练并保存模型
    #######################################################
    if is_need_train:
        train_and_save_weights()
    #######################################################
    # 加载已经保存的模型
    #######################################################
    model = MyNetwork()
    model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
    model.load_weights(save_weights_path)
    print('loaded weights from file.')
    model.evaluate(db_test)







