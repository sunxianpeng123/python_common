# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_tensorboard.py
@time: 2020/1/5 17:08
"""


import datetime
import io
import os
from matplotlib import pyplot as plt

import tensorflow as tf
from keras import datasets, layers, optimizers, Sequential, metrics
import datetime
from keras.callbacks import TensorBoard

"""
1、cmd

"""
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

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def image_grid(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return figure

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



def train(db_train, db_test, model, epochs, optimizer,logs_dir):
    """模型训练"""
    """可视化相关代码"""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logs_dir = logs_dir + current_time
    print('logs_dir = {}'.format(logs_dir))
    summary_writer = tf.summary.create_file_writer(logs_dir)
    # get x from (x,y)
    sample_img = next(iter(db_train))[0]
    # get first image instance
    sample_img = sample_img[0]
    sample_img = tf.reshape(sample_img, [1, 28, 28, 1])
    """可视化图片"""
    with summary_writer.as_default():
        tf.summary.image("Training sample:", sample_img, step=0)
        # 可视化多个图片
        # tf.summary.image('multi images:', images,max_outputs=25,step=0)
    """ """
    # val_images = tf.reshape(val_images,[-1, 28, 28])
    # figure = image_grid(val_images)
    # tf.summary.image('val-images:',plot_to_image(figure),step=step)
    # 算法迭代
    for epoch in range(epochs):
        # 训练
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
            grads = tape.gradient(loss_cross_entropy, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_cross_entropy), float(loss_mse))
            """可视化标量"""
            with summary_writer.as_default():
                tf.summary.scalar('train loss', float(loss_cross_entropy),step=epoch)
        # 验证
        total_correct = 0
        total_num = 0
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
        accuracy = total_correct / total_num
        print(epoch, 'test acc:', accuracy)
        """可视化标量"""
        with summary_writer.as_default():
            tf.summary.scalar('test accuracy', float(accuracy),step=epoch)

if __name__ == '__main__':
    """mnist手写字体识别"""
    batch_size = 128
    learning_rate = 1e-3
    epochs = 30
    logs_dir = 'logs/'
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
    train(db_train,db_test,model,epochs,optimizer,logs_dir)