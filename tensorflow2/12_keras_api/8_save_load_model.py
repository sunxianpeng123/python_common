# -*- coding: utf-8 -*-
# @Time : 2020/1/6 23:41
# @Author : sxp
# @Email : 
# @File : 5_save_load_model.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow import keras
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    """数据转换, x is a image, not a batch"""
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

def check_data(db):
    db_iter = iter(db)
    sample = next(db_iter)
    # 打印一个批次的维度信息 和 批次大小
    print('batch:',sample[0].shape, sample[1].shape)
#################################################################
#自定义层
#################################################################
class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(MyDense, self).__init__()
        # 不可以使用tf.constant
        self.kernel = self.add_weight('w', [input_dim, output_dim])
        self.bias = self.add_weight('b', [output_dim])
    #__call__
    def call(self, inputs, training=None):
        """
        :param inputs: 输入的数据
        :param training:指定当前的forward是training还是test
        :return:
        """
        out = inputs @ self.kernel + self.bias
        return out

#################################################################
#自定义模型
#################################################################
class MyModel(keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = MyDense(28 * 28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)

    def call(self,inputs, training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x
#################################################################
# 使用训练好的模型预测
#################################################################
def predict(db_test, model):
    """对一个批次样本进行预测"""
    sample = next(iter(db_test))
    x = sample[0]#一个批次的图片
    y = sample[1]#一个批次的图片onehot
    pred = model.predict(x)#[b,10]
    print('x.shape={},y.shape={},pred.shape={}'.format(x.shape, y.shape, pred.shape))
    # convert back to number
    y = tf.argmax(pred, axis=1)
    print(pred)
    print(y)
    print("====================")
    test_x = sample[0]#一个批次的图片
    test_y = sample[1]#一个批次的图片onehot
    test_x = tf.expand_dims(test_x[0,:], axis=0)
    test_y = tf.expand_dims(test_y[0,:], axis=0)
    test_pred = model.predict(test_x)
    test_pred = tf.argmax(test_pred, axis=1)
    print('test_x.shape={},test_y.shape={},test_pred.shape={}'.format(test_x.shape, test_y.shape, test_pred.shape))
    print('test_y = {}, test pred = {}'.format(test_y, test_pred))
#################################################################
# 保存和加载模型
#################################################################
def save_model(model, save_model_path):
    """保存模型"""
    if os.path.exists(save_model_path) == False:
        os.makedirs(save_model_path)
    # 报错
    # NotImplementedError: The `save` method requires the model to be a Functional model or a Sequential model.
    # It does not work for subclassed models, because such models are defined via the body of a Python method,
    # which isn't safely serializable. Consider using `save_weights`, in order to save the weights of the model.
    model.save(save_model_path)
    print('save model done!')

def load_model():
    """加载保存的模型"""
    pass


if __name__ == '__main__':
    batch_size = 128
    learning_rate = 1e-3
    epochs = 1
    save_model_dir = 'model/'
    load_model_dir = save_model_dir
    print(os.path.abspath(save_model_dir))


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
    model = MyModel()

    optimizer = optimizers.Adam(lr=learning_rate)
    # 指定训练集的优化函数，损失函数，测量尺
    model.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # 指定训练集，迭代次数epochs，验证集，测试集集频率（即每迭代几次做一次模型验证,会打印相关信息，用于停止、保存等操作）
    model.fit(db_train, epochs=epochs, validation_data=db_test, validation_freq=1)

    print("########################")
    # 模型验证
    model.evaluate(db_test)
    # 预测
    predict(db_test,model)
    #################################################################
    # 保存和加载模型
    #################################################################
    """保存模型"""
    save_model_path = save_model_dir + str(time.time()).replace('.', '')
    save_model(model, save_model_path)
    """加载保存的模型"""
    print("model_loaded########################")
    time.sleep(10)
    load_model_path = save_model_path
    model_loaded = tf.keras.models.load_model(load_model_path, compile=False)
    # 指定训练集的优化函数，损失函数，测量尺
    model_loaded.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # 指定训练集，迭代次数epochs，验证集，测试集集频率（即每迭代几次做一次模型验证,会打印相关信息，用于停止、保存等操作）
    model_loaded.fit(db_train, epochs=epochs, validation_data=db_test, validation_freq=1)
    model_loaded.evaluate(db_test)

