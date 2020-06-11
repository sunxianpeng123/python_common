# encoding: utf-8

"""
@author: sunxianpeng
@file: detect_server.py
@time: 2020/6/11 10:22
"""

import time
from utils import get_logger, get_this_file_name
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

from utils.yolo_utils import scale_boxes, read_classes, read_anchors, preprocess_image, generate_colors, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from keras import backend as K

##############################################################
# Log
##############################################################
log_path = 'logs/'
# today_datetime = time.strftime("%Y%m%d%H%M%S", time.localtime())
today_datetime = time.strftime("%Y%m%d", time.localtime())
log_name = log_path + 'logs_' + str(today_datetime) + '.log'
log_file = log_name
this_file_name = get_this_file_name()
logger = get_logger(log_file, this_file_name)
logger.info('start')

# 过滤掉那些概率低的边框
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """
    过滤掉概率小的边框
    参数:
    box_confidence -- 装载着每个边框的pc，维度是(19, 19, 5, 1)
    boxes -- 装载着每个边框的bx,by,bh,bw，维度是(19, 19, 5, 4)
    box_class_probs -- 装载着每个边框的80个种类的概率，维度是(19, 19, 5, 80)
    threshold -- 阈值，概率低于这个值的边框会被过滤掉
    返回值:
    scores -- 装载保留下的那些边框的概率，维度是 (None,)
    boxes -- 装载保留下的那些边框的坐标(b_x, b_y, b_h, b_w)，维度是(None, 4)
    classes -- 装载保留下的那些边框的种类的索引，维度是 (None,)
    为什么返回值中的维度都是None呢？因为我们不知道会有多少边框被过滤掉，有多少会被保留下来。
    """
    # 将px和c相乘
    box_scores = np.multiply(box_confidence, box_class_probs)
    # 因为之前我们把keras导入成了K，所以下面是调用得keras的函数
    box_classes = K.argmax(box_scores, axis=-1)  # 获取概率最大的那个种类的索引
    box_class_scores = K.max(box_scores, axis=-1)  # 获取概率最大的那个种类的概率值
    # 采集一个过滤器。当某个种类的概率值大于等于阈值threshold时，
    # 对应于这个种类的filtering_mask中的位置就是true，否则就是false。
    # 所以filtering_mask就是[False, True, 。。。, False, True]这种形式。
    filtering_mask = K.greater_equal(box_class_scores, threshold)
    # 用上面的过滤器来过滤掉那些概率小的边框。
    # 过滤完成后，scores和boxes，classes里面就只装载了概率大的边框的概率值和坐标以及种类索引了。
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    return scores, boxes, classes

with tf.compat.v1.Session() as test_a:
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = 0.5)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))


# 用非最大值抑制技术过滤掉重叠的边框
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    参数:
    scores -- 前面yolo_filter_boxes函数保留下的那些边框的概率值，维度是 (None,)
    boxes -- 前面yolo_filter_boxes函数保留下的那些边框的坐标(b_x, b_y, b_h, b_w)，维度是(None, 4)
    classes -- 前面yolo_filter_boxes函数保留下的那些边框的种类的索引，维度是 (None,)
    max_boxes -- 最多想要保留多少个边框
    iou_threshold -- 交并比，这是一个阈值，也就是说交并比大于这个阈值的边框才会被进行非最大值抑制处理

    Returns:
    scores -- NMS保留下的那些边框的概率值，维度是 (, None)
    boxes -- NMS保留下的那些边框的坐标，维度是(4, None)
    classes -- NMS保留下的那些边框的种类的索引，维度是(, None)
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')

    # tf.compat.v1.keras.backend.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    # tf.compat.v1.keras.backend.get_session().run(tf.Variable.initializer([max_boxes_tensor]))#.set_session(tf.compat.v1.Session(config=config))
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

    # tensorflow为我们提供了一个NMS函数，我们直接调用就可以了tf.image.non_max_suppression()。
    # 这个函数会返回NMS后保留下来的边框的索引
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)

    # 通过上面的索引来分别获取被保留的边框的相关概率值，坐标以及种类的索引
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

with tf.compat.v1.Session() as test_b:
    scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))


# 这个函数里面整合了前面我们实现的两个过滤函数。将YOLO模型的输出结果输入到这个函数后，这个函数会将多余边框过滤掉

def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    参数:
    yolo_outputs -- YOLO模型的输出结果
    image_shape -- 图像的原始像素。因为网上的YOLO需要的图像像素是608，而我们的原始图像是720*1280的。
                   所以在输入到YOLO前我们会转成608的；同理，我们也会将YOLO的输出结果转回到720*1280。
    max_boxes -- 你希望最多识别出多少个边框
    score_threshold -- 概率值阈值
    iou_threshold -- 交并比阈值

    Returns:
    scores -- 最终保留下的那些边框的概率值，维度是 (, None)
    boxes -- 最终保留下的那些边框的坐标，维度是(4, None)
    classes -- 最终保留下的那些边框的种类的索引，维度是(, None)
    """

    ### 将YOLO输出结果分成3份，分别表示概率值，坐标，种类索引
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 因为YOLO模型使用的坐标表示法是(x,y,w,h),所以我们先要将其转成(x1, y1, x2, y2)的形式
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 使用我们前面实现的yolo_filter_boxes函数过滤掉概率值低于阈值的边框
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    # 将（608，608）转回为（720，1280）
    boxes = scale_boxes(boxes, image_shape)

    # 使用我们前面实现的yolo_non_max_suppression过滤掉重叠的边框
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes,
                                                      iou_threshold=iou_threshold)

    return scores, boxes, classes

with tf.compat.v1.Session() as test_b:
    yolo_outputs = (tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = yolo_eval(yolo_outputs)
    print("scores[2] = " + str(scores[2].eval()))
    print("boxes[2] = " + str(boxes[2].eval()))
    print("classes[2] = " + str(classes[2].eval()))
    print("scores.shape = " + str(scores.eval().shape))
    print("boxes.shape = " + str(boxes.eval().shape))
    print("classes.shape = " + str(classes.eval().shape))

"""使用YOLO模型进行车辆探测"""
sess = K.get_session()
# 定义种类以及anchor box和像素
# 我们已经将种类和anchor box的信息保存在了"coco_classes.txt" 和 "yolo_anchors.txt"文件中了。所以从它们里面读取出来就可以了。
# 我们的原始图像像素是(720., 1280.)
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)
# 加载已经训练好了的YOLO模型
# 如何执行这句时出现错误提示“The kernel appears to have died. It will restart automatically”，那么你就要重新生成与自己开发环境匹配的yolo.h5。
# 步骤如下：
# 1，将本文档目录下的YAD2Kmaster文件夹复制到C盘下
# 2，打开Anaconda prompt
# 3，执行activate tensorflow命令
# 4，执行cd C:\YAD2Kmaster命令
# 5，执行python yad2k.py yolov2.cfg yolov2.weights model_data/yolo.h5命令
# 6，用C盘的YAD2Kmaster文件夹下的model_data替换掉本文档目录下的model_data
yolo_model = load_model("model_data/yolo.h5")
# 下面的代码显示出了这个YOLO模型每一层网络的信息
yolo_model.summary()
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
#  将YOLO模型的输出结果转换成我们需要的格式
# 因为 yolo_model 输出的维度是(m, 19, 19, 5, 85)，所以我们要转换一下，这样好满足我们的yolo_eval函数的输入格式要求。
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
# 过滤边框
# yolo_model输出的yolo_outputs里面包含了它探测到的所有的边框。下面我们将使用我们自己实现的过滤函数yolo_eval来过滤掉多余的边框。
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
# 开始探测图片
# 使用前面构建好的graph来探测图片中的车辆
def predict(sess, image_file):
    """
    参数:
    sess -- 前面我们创建的包含了YOLO模型和过滤函数的graph
    image_file -- 待探测的图像
    """
    # 将图片读取出来，并且转换成608像素
    image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    # 运行我们之前构建好的graph
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    # 打印出找到了几个边框
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # 为种类分配颜色
    colors = generate_colors(class_names)
    # 将找到的边框在图片上画出来
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    # 将画出边框的图片显示出来
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict(sess, "test.jpg")






