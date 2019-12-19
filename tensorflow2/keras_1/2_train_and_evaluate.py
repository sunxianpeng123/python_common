# -*- coding: utf-8 -*-
# @Time : 2019/12/20 0:10
# @Author : sxp
# @Email : 
# @File : 2_train_and_evaluate.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

"""3、训练和评估"""

def train_flow():
    """3.1设置训练流程"""
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    return model

def input_numpy_data():
    train_x = np.random.random((1000,72))
    train_y = np.random.random((1000,10))

    val_x = np.random.random((200,72))
    val_y = np.random.random((200,10))
    return train_x,train_y,val_x,val_y

def input_tf_data(train_x, train_y, val_x, val_y):
    """3.3 tf.data输入数据"""
    dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    dataset = dataset.shuffle(2)  # 将数据打乱，数值越大，混乱程度越大
    dataset = dataset.batch(32)## 按照顺序取出4行数据，最后一次输出可能小于batch
    # 数据集重复了指定次数,# repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
    # 为了配合输出次数，一般默认repeat()空
    dataset = dataset.repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y))
    val_dataset = val_dataset.batch(32)
    val_dataset = val_dataset.repeat()
    return dataset,val_dataset

def test_data():
    """3.4 评估与预测"""
    test_x = np.random.random((1000,72))
    test_y = np.random.random((1000,10))
    test_data_tf = tf.data.Dataset.from_tensor_slices((test_x,test_y))
    test_data_tf = test_data_tf.batch(32).repeat()

    return test_x,test_y,test_data_tf

if __name__ == '__main__':
    test_x,test_y,test_data_tf = test_data()
    train_x, train_y, val_x, val_y = input_numpy_data()
    """1、numpy data"""
    model_np = train_flow()
    model_np.fit(train_x,train_y,epochs=10,batch_size=100,validation_data=(val_x,val_y))
    model_np.evaluate(test_x,test_y,batch_size=32)
    predict_np = model_np.predict(test_x,batch_size=32)
    print('predict_np = {}'.format(predict_np))

    """2、tf data"""
    model_tf = train_flow()
    dataset, val_dataset = input_tf_data(train_x, train_y, val_x, val_y)
    # steps_per_epoch表示是将一个epoch分成多少个batch_size， 如果训练样本数N=1000，steps_per_epoch = 10，那么相当于一个batch_size=100
    # validation_steps表示将验证集分成的batch数，每个batch中数据个数是 验证集总数/validation_steps
    model_tf.fit(dataset,epochs=10,steps_per_epoch=30,validation_data=val_dataset,validation_steps=3)
    model_tf.evaluate(test_data_tf,steps=30)
    predict_tf = model_tf.predict(test_data_tf,steps=30)
    print('predict_tf = {}'.format(predict_tf))
