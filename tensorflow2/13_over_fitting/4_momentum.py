# -*- coding: utf-8 -*-
# @Time : 2020/1/8 0:29
# @Author : sxp
# @Email : 
# @File : 1_train_val_test.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import datasets,layers,optimizers,Sequential,metrics
from tensorflow import keras
from tensorflow.keras import regularizers
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
# model
#################################################################
def get_model():
    model = Sequential([layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
                          layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
                          layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
                          layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
                          layers.Dense(10)])
    model.build(input_shape=(None, 28 * 28))
    # model.summary()
    return model

if __name__ == '__main__':
    batch_size = 128
    learning_rate = 1e-3
    epochs = 1
    save_weights_dir = 'model/weights/'
    load_weights_dir = save_weights_dir
    mnist_path =os.path.abspath(r'../data/mnist.npz')

    print(os.path.abspath(save_weights_dir))
    print('mnist path = {}'.format(mnist_path))

    (x, y), (x_test, y_test) = datasets.mnist.load_data(mnist_path)
    print('x shape = {},y shape = {}'.format(x.shape, y.shape))

    """划分训练集、验证集、测试集"""
    idx = tf.range(60000)
    idx = tf.random.shuffle(idx)
    x_train, y_train = tf.gather(x, idx[:50000]),tf.gather(y,idx[:50000])
    x_val, y_val = tf.gather(x, idx[-10000:]), tf.gather(y, idx[-10000:])
    print("x_train.shape, y_train.shape, x_val.shape, y_val.shape = ",x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(preprocess).shuffle(10000).batch(batch_size)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(preprocess).shuffle(10000).batch(batch_size)
    db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).map(preprocess).batch(batch_size)
    # check_data(db_train)
    sample = next(iter(db_train))
    print(sample[0].shape, sample[1].shape)
    """模型相关"""
    model = get_model()
    # 以下可以加上momentum参数
    # optimizer = optimizers.SGD(lr=learning_rate,momentum=0.9)
    optimizer = optimizers.RMSprop(lr=learning_rate, momentum=0.9)
    #下面个为啥报错？？？ TypeError: Unexpected keyword argument passed to optimizer: beta_1
    # optimizer = optimizers.SGD(lr=learning_rate,beta_1=0.9,beta_2=0.999)
    # 指定训练集的优化函数，损失函数，测量尺
    model.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    # 指定训练集，迭代次数epochs，验证集，测试集集频率（即每迭代几次做一次模型验证,会打印相关信息，用于停止、保存等操作）
    model.fit(db_train, epochs=epochs, validation_data=db_val, validation_freq=1)

    print('Test performance:')
    model.evaluate(db_test)
    # predict(db_test,model)