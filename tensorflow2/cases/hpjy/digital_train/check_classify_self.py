# encoding: utf-8

"""
@author: sunxianpeng
@file: check_classify.py
@time: 2020/6/3 15:13
"""
import logging
import uuid

import numpy as np
import cv2
from skimage import measure
import os
import tensorflow as tf
import matplotlib.pyplot as plt


def lap_filter2(imgs, k_max=6):
    result = []
    for i in imgs:

        img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        if np.sum(img) < 0.1:
            result.append(f1)
        else:
            kernel_lap = np.array([[0, -1, 0],
                                   [-1, k_max, -1],
                                   [0, -1, 0]])
            f1 = cv2.filter2D(img, -1, kernel_lap)
            fz = 200
            f1[f1 < fz] = 0
            f1[f1 >= fz] = 1
            result.append(f1)
    return result


def classify_gray(model, imgs):
    image_shape = (28, 28, 1)
    predict_image_set = []
    for i in imgs:
        img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, image_shape[:2])
        data = np.array(img).reshape(image_shape[0], image_shape[1], 1)
        predict_image_set.append(data)

    pd = np.array(predict_image_set)
    rs = model.predict(pd)
    result = np.argmax(rs, axis=1)
    # print (rs,result)
    r_no = 0
    returnList = []
    for r in result:
        score = rs[r_no][r]
        print("得分:", score, r)
        if score < 0.8:
            returnList.append(-1)
        else:
            returnList.append(r)
        r_no = r_no + 1

    return returnList


def kill_classify(images):
    result = []
    showImage(images)
    # images_2z=lap_filter2(images)
    images_2z = images
    result = classify_gray(model1, images_2z)

    return result

def showImage(images):
    plt.figure()
    img_index = 1
    for img in images:
        plt.subplot(1, 10, img_index)
        plt.imshow(img)
        img_index = img_index + 1
    plt.show()




model_name='model/model_self.h5'
model_path=model_name
logging.info("start")
model1 = tf.keras.models.load_model(model_path)
image_shape = (28,28, 1)
logging.info("end")

test_path = "val/8"
image_paths = [os.path.join(test_path, i) for i in list(os.listdir(test_path))]
no = 1
for img_path in image_paths:
    if img_path.endswith(".jpg"):
        if no % 11 == 0:
            print(img_path, kill_classify([cv2.imread(img_path)]))
        no = no + 1

