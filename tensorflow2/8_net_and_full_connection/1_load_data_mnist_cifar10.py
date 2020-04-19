# -*- coding: utf-8 -*-
# @Time : 2020/1/1 23:11
# @Author : sxp
# @Email : 
# @File : 1_load_data_mnist_cifar10.py
# @Project : python_common


import tensorflow as tf
from tensorflow.keras import datasets
from matplotlib import pyplot as plt
import os


def show_img(img):
    plt.imshow(img)
    plt.show()

def tf_mnist():
    """加载 mnist数据集，及查看 mnist 数据集的信息"""
    print("########################1、tf_mnist############################")
    # mnist_path = r'F:\PythonProjects\python_common\tensorflow2\data\mnist.npz'
    mnist_path = os.path.abspath(r'../data/mnist.npz')
    # exit(0)
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data(mnist_path)
    print('x_train shape = {},y_train shape = {}'.format(x_train.shape,y_train.shape))
    # x中值得最小、最大和平均值
    print('x min = {}, x max = {},x mean = {}'.format(x_train.min(),x_train.max(),x_train.mean()))
    # 还是numpy的格式
    print('x train type = {}, y train type = {}'.format(type(x_train),type(y_train)))
    #查看前四个标签
    print('y train = {}'.format(y_train[:3]))
    # one_hot 编码
    y_train_onehot = tf.one_hot(y_train,depth=10)
    print(y_train_onehot[:2])

    return None

def tf_cifar_10_100():
    """加载 cifar_10_100，及查看 cifar_10_100 数据集的信息"""
    print("########################2、tf_cifar_10_100 ############################")
    (x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
    print('x_train shape = {},y_train shape = {}'.format(x_train.shape,y_train.shape))
    # x中值得最小、最大和平均值
    print('x min = {}, x max = {},x mean = {}'.format(x_train.min(),x_train.max(),x_train.mean()))
    # 还是numpy的格式
    print('x train type = {}, y train type = {}'.format(type(x_train),type(y_train)))
    #查看前四个标签
    print('y train = {}'.format(y_train[:3]))
    # one_hot 编码
    y_train_onehot = tf.one_hot(y_train,depth=10)
    print(y_train_onehot[:2])

    return None

def tf_data_dataset():
    """转成Dataset对象，迭代操作"""
    print("########################3、tf_data_dataset ############################")
    (x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
    print("只操作图片=============")
    # 将 numpy 数据格式转换成 TensorFlow tensor格式
    db_1 =tf.data.Dataset.from_tensor_slices(x_test)
    # 取出测试集中的一张图片，然后显示
    x_test_one_img_1 = next(iter(db_1))
    # show_img(x_test_one_img_1)
    print('x_test_one_img_1 shape = {}'.format(x_test_one_img_1.shape))
    print("同时操作图片和标签=============")
    # 将 numpy 数据格式转换成 TensorFlow tensor格式
    db_2 =tf.data.Dataset.from_tensor_slices((x_test,y_test))
    # 取出测试集中的一张图片，然后显示
    x_test_one_img_2,x_test_one_label_2 = next(iter(db_2))
    show_img(x_test_one_img_2)
    print('x_test_one_img_2 shape = {},x_test_one_label_2 = {}'
          .format(x_test_one_img_2.shape,x_test_one_label_2))

    return None



if __name__ == '__main__':
    # tf_mnist()
    tf_cifar_10_100()
    # tf_data_dataset()