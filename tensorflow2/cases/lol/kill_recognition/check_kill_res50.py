# encoding: utf-8

"""
@author: sunxianpeng
@file: check_kill_res50.py
@time: 2020/6/4 10:58
"""
import cv2
import numpy as np
import tensorflow as tf

import os
global ok_count, total_count
ok_count = 0
total_count = 0

def classify_gray2(model, imgs):
    global ok_count, total_count
    predict_image_set = []
    for i in imgs:
        img = i
        img = cv2.resize(img, image_shape[:2])
        data = np.array(img).reshape(image_shape[0], image_shape[1], 3)
        predict_image_set.append(data)

    pd = np.array(predict_image_set)
    pd = tf.cast(pd, tf.float32)
    rs = model.predict(pd)
    result = np.argmax(rs, axis=1)
    # print (rs,result)
    r_no = 0
    returnList = []
    for r in result:
        total_count = total_count + 1
        score = rs[r_no][r]
        print(r, "得分:", score)
        if score < 0.8:
            returnList.append(-1)
        else:
            returnList.append(r)
            if (r == 1):
                ok_count = ok_count + 1
        r_no = r_no + 1

    return returnList


test_path = "val/0"
image_shape = (32, 32, 3)

image_paths = [os.path.join(test_path, i) for i in list(os.listdir(test_path))]

model_path="model/kill_dead_model_res.h5"
model = tf.keras.models.load_model(model_path)

for img_path in image_paths:
    if img_path.endswith(".jpg"):
        iimmg = cv2.imread(img_path)
        result = classify_gray2(model, [iimmg])
        print(result)

print(ok_count / total_count)