# encoding: utf-8

"""
@author: sunxianpeng
@file: check_blood_res50.py
@time: 2020/6/4 11:44
"""
import cv2
import numpy as np
import tensorflow as tf

import os

from blood_recognition import read_img_with_chinese_path_or_not

global ok_count, total_count

def classify_gray2(model,imgs):
    image_shape = (32, 32, 1)
    predict_image_set=[]
    for i in imgs:
        img=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,image_shape[:2])
        data = np.array(img).reshape(image_shape[0],image_shape[1],1)
        predict_image_set.append(data)

    pd=np.array(predict_image_set)
    pd = tf.cast(pd, tf.float32)
    rs = model.predict(pd)
    result = np.argmax(rs, axis=1)
    #print (rs,result)
    r_no=0
    returnList=[]
    for r in result :
        score=rs[r_no][r]
        print ("得分:",score)
        if score<0.9:
            returnList.append(-1)
        else:
            returnList.append(r)
        r_no=r_no+1

    return returnList

test_path = "val/0"
image_paths = [os.path.join(test_path,i) for i in list(os.listdir(test_path))]

model_path="model/model_tiny_final.h5"
model = tf.keras.models.load_model(model_path)

for img_path in image_paths:
    if img_path.endswith(".jpg",0):
        iimmg = read_img_with_chinese_path_or_not(img_path)
        result=classify_gray2(model,[iimmg])
        print(result)