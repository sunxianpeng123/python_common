import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
print(tf.__version__)


def tf_clip_by_global_norm():
    """ global_norm 是所有梯度的平方和
        Gradient Clipping的引入是为了处理 gradient explosion (梯度爆炸) 或者 gradients vanishing （梯度消失）的问题。当在一次迭代中权重的更新过于迅猛的话，
        很容易导致 loss divergence。Gradient Clipping (梯度裁剪)的直观作用就是让权重的更新限制在一个合适的范围。
        具体的细节是
        １．在solver中先设置一个clip_gradient
        ２．在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，
            而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > clip_gradient，
            则求缩放因子scale_factor = clip_gradient / sumsq_diff。这个scale_factor在(0,1)之间。
            如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。
        ３．最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。
        这样就保证了在一次迭代更新中，所有权重的梯度的平方和在一个设定范围以内，这个范围就是clip_gradient.
    """
    # 784 => 512
    w1, b1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1)), tf.Variable(tf.zeros([512]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(lr=0.01)
    for step, (x,y) in enumerate(train_db):
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 784))
        with tf.GradientTape() as tape:
            # layer1.
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            # layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # output
            out = h2 @ w3 + b3
            # out = tf.nn.relu(out)

            # compute loss
            # [b, 10] - [b, 10]
            loss = tf.square(y-out)
            # [b, 10] => [b]
            loss = tf.reduce_mean(loss, axis=1)
            # [b] => scalar
            loss = tf.reduce_mean(loss)

        # compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # print('==before==')
        # for g in grads:
        #     print(tf.norm(g))

        grads,  _ = tf.clip_by_global_norm(grads, 15)

        # print('==after==')
        # for g in grads:
        #     print(tf.norm(g))
        # update w' = w - lr*grad
        optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2, w3, b3]))
        if step % 100 == 0:
            print(step, 'loss:', float(loss))

if __name__ == '__main__':
    # (x, y), _ = datasets.mnist.load_data()
    data_dir = r'E:\PythonProjects\python_common\tensorflow2\data\mnist.npz'
    (x, y), _ = datasets.mnist.load_data(data_dir)
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 50.
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=10)
    print('x:', x.shape, 'y:', y.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128).repeat(30)
    x, y = next(iter(train_db))
    print('sample:', x.shape, y.shape)
    # print(x[0], y[0])
    main()