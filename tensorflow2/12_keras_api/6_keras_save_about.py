# -*- coding: utf-8 -*-
# @Time : 2020/1/7 0:41
# @Author : sxp
# @Email : 
# @File : 5_t.py
# @Project : python_common

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,models,callbacks
from tensorflow.keras.datasets import mnist
import os
import time

def preprocess(x, y):
    """数据转换, x is a image, not a batch"""
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x,y

def create_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', metrics=['accuracy'],
                  loss='sparse_categorical_crossentropy')

    return model

def evaluate(target_model,test_x,test_y):
    _, acc = target_model.evaluate(test_x, test_y)
    print("Restore model, accuracy: {:5.2f}%".format(100*acc))


######################################################
#自动保存 checkpoints
######################################################
def checkpoint():
    """自动保存 checkpoints
    这样做，一是训练结束后得到了训练好的模型，使用得不必再重新训练，二是训练过程被中断，可以从断点处继续训练。
    设置tf.keras.callbacks.ModelCheckpoint回调可以实现这一点。
    """
    # 存储模型的文件名，语法与 str.format 一致
    # period=10：每 10 epochs 保存一次
    checkpoint_path = "chechpoint/"+ str(time.time()).replace('.','') +"/training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=10)

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(x_train, y_train, epochs=50, callbacks=[cp_callback],
              validation_data=(x_test, y_test), verbose=0)
    """加载权重"""
    print('=================加载权重=================')
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    # 'training_2/cp-0050.ckpt'
    model = create_model()
    model.load_weights(latest)
    evaluate(model, x_test, y_test)

def mannul_save_weights():
    """手动保存权重"""
    # 存储模型的文件名，语法与 str.format 一致
    # period=10：每 10 epochs 保存一次
    checkpoint_path = "chechpoint/"+ str(time.time()).replace('.','') +"/training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=10)

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(x_train, y_train, epochs=50, callbacks=[cp_callback],
              validation_data=(x_test, y_test), verbose=0)
    """保存权重"""
    model.save_weights('./checkpoints/mannul_checkpoint')
    """加载权重"""
    print('=================加载权重=================')
    model = create_model()
    model.load_weights('./checkpoints/mannul_checkpoint')
    evaluate(model, x_test, y_test)


def save_model_HDF5():
    """
    保存整个模型
    上面的示例仅仅保存了模型中的权重(weights)，模型和优化器都可以一起保存，包括权重(weights)、模型配置(architecture)和
    优化器配置(optimizer configuration)。这样做的好处是，当你恢复模型时，完全不依赖于原来搭建模型的代码。
    保存完整的模型有很多应用场景，比如在浏览器中使用 TensorFlow.js 加载运行，比如在移动设备上使用 TensorFlow Lite 加载运行。
    :return:
    """
    # 存储模型的文件名，语法与 str.format 一致
    # period=10：每 10 epochs 保存一次
    checkpoint_path = "chechpoint/"+ str(time.time()).replace('.','') +"/training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=10)

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(x_train, y_train, epochs=50, callbacks=[cp_callback],
              validation_data=(x_test, y_test), verbose=0)
    """保存整个模型"""
    # 直接调用model.save即可保存为 HDF5 格式的文件。
    model_path = 'model/' + str(time.time()).replace('.', '') + '.h5'
    model.save(model_path)
    """加载权重"""
    print('=================加载权重=================')
    model = models.load_model(model_path)
    evaluate(model, x_test, y_test)


def save_model_saved_model():
    """保存为saved model"""
    # 存储模型的文件名，语法与 str.format 一致
    # period=10：每 10 epochs 保存一次
    checkpoint_path = "chechpoint/"+ str(time.time()).replace('.','') +"/training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=10)

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(x_train, y_train, epochs=50, callbacks=[cp_callback],
              validation_data=(x_test, y_test), verbose=0)
    """保存为saved_model格式。"""
    saved_model_path = "./saved_models/{}".format(int(time.time()))
    tf.keras.experimental.export_saved_model(model, saved_model_path)
    print('=================恢复模型并预测=================')
    new_model = tf.keras.experimental.load_from_saved_model(saved_model_path)
    print(model.predict(x_test).shape)
    """
    saved_model格式的模型可以直接用来预测(predict)，但是 saved_model 没有保存优化器配置，
    如果要使用evaluate方法，则需要先 compile。
    """
    new_model.compile(optimizer=model.optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    evaluate(model, x_test, y_test)



if __name__ == '__main__':
    batch_size = 128
    learning_rate = 1e-3
    epochs = 1

    mnist_path = os.path.abspath(r'../data/mnist.npz')
    print('mnist path = {}'.format(mnist_path))

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(mnist_path)
    print('x train shape = {},y train shape = {}'.format(x_train.shape, y_train.shape))

    # db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    #     .map(preprocess).shuffle(10000).batch(batch_size)
    # db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
    #     .map(preprocess).batch(batch_size)

    y_train, y_test = y_train[:1000], y_test[:1000]
    x_train = x_train[:1000].reshape(-1, 28 * 28) / 255.0
    x_test = x_test[:1000].reshape(-1, 28 * 28) / 255.0
    # 自动保存权重
    # checkpoint()
    # 手动保存权重
    # mannul_save_weights()
    # 保存整个模型HDF5
    # save_model_HDF5()
    # saved_model
    save_model_saved_model()
