# -*- coding: utf-8 -*-
# @Time : 2020/1/2 1:04
# @Author : sxp
# @Email : 
# @File : 2_data_preprocess.py
# @Project : python_common

import tensorflow as tf
from tensorflow.keras import datasets
import os

def tf_shuffle():
    """shuffle的功能为打乱dataset中的元素，它有一个参数buffersize，
    表示打乱时使用的buffer的大小，建议舍的不要太小，一般是1000："""
    print("########################1、tf_shuffle ############################")
    data_dir =os.path.abspath(r'../data/mnist.npz')
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(data_dir)
    db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    db =db.shuffle(10000)
    return None

def tf_map():
    """和python中的map类似，map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset，
        注意map函数可以使用num_parallel_calls参数加速
    """
    print("########################2、tf_map ############################")
    data_dir =os.path.abspath(r'../data/mnist.npz')
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(data_dir)
    db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    # 处理过程
    def preprocess(img,label):
        """对每一张图片和其标签进行处理"""
        img =tf.cast(img, dtype=tf.float32) / 255.
        label = tf.cast(label, dtype=tf.int32)
        label = tf.one_hot(label, depth=10)
        return img,label

    db = db.map(preprocess)

    res = next(iter(db))
    # 取出一张图片和标签，并查看对应的tensor形状
    print('img shape = {}, label shape = {}'.format(res[0].shape, res[1].shape))
    # 查看一张图片的onehot编码
    print('label one hot value = {}'.format(res[1][:100]))
    # img shape = (32, 32, 3), label shape = (1, 10)
    # [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]
    return None

def tf_batch():
    """batch就是将多个元素组合成batch，按照输入元素第一个维度"""
    print("########################3、tf_batch ############################")
    data_dir =os.path.abspath(r'../data/mnist.npz')
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(data_dir)
    db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    # 处理过程
    def preprocess(img,label):
        """对每一张图片和其标签进行处理"""
        img =tf.cast(img, dtype=tf.float32) / 255.
        label = tf.cast(label, dtype=tf.int32)
        label = tf.one_hot(label, depth=10)
        # 上面的 label形状是（1,10），添加batch维度之后，需要将 1 去掉
        label = tf.squeeze(label)
        return img,label

    db = db.map(preprocess).batch(32)
    res = next(iter(db))
    # 取出一张图片和标签，并查看对应的tensor形状
    print('img shape = {}, label shape = {}'.format(res[0].shape, res[1].shape))
    return None

def tf_repeat():
    """repeat 方法在读取到组后的数据时重启数据集。
    要限制epochs的数量，可以设置count参数。
    为了配合输出次数，一般默认repeat()空,即无限次"""
    print("########################4、tf_repeat #########################")
    data_dir =os.path.abspath(r'../data/mnist.npz')
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(data_dir)
    db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    # 处理过程
    def preprocess(img,label):
        """对每一张图片和其标签进行处理"""
        img =tf.cast(img, dtype=tf.float32) / 255.
        label = tf.cast(label, dtype=tf.int32)
        label = tf.one_hot(label, depth=10)
        # 上面的 label形状是（1,10），添加batch维度之后，需要将 1 去掉
        label = tf.squeeze(label)
        return img,label

    epochs = 10
    db = db.map(preprocess).batch(32).repeat(epochs)
    return None

def full_data_preprocess_example():
    print("###################5、full_data_preprocess_example ################")
    data_dir =os.path.abspath(r'../data/mnist.npz')
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(data_dir)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    # 处理过程
    def preprocess(img,label):
        """对每一张图片和其标签进行处理"""
        img =tf.cast(img, dtype=tf.float32) / 255.
        label = tf.cast(label, dtype=tf.int32)
        label = tf.one_hot(label, depth=10)
        # 上面的 label形状是（1,10），添加batch维度之后，需要将 1 去掉
        label = tf.squeeze(label)
        return img,label

    # epochs = 10
    ds_train = ds_train.map(preprocess).shuffle(10000).batch(32)
    ds_test = ds_test.map(preprocess).shuffle(10000).batch(32)
    return None


if __name__ == '__main__':
    tf_shuffle()
    tf_map()
    tf_batch()
    full_data_preprocess_example()