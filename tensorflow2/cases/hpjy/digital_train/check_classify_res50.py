# encoding: utf-8

"""
@author: sunxianpeng
@file: check_classify.py
@time: 2020/6/3 15:13
"""
import uuid

import numpy as np
import cv2
from skimage import measure
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def lap_filter1(imgs, k_max=5):
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
            # print (f1)
            fz = 240
            f1[f1 < fz] = 0
            f1[f1 >= fz] = 1
            result.append(f1)
    return result

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

def block_extract_t(gray_image):
    result = measure.label(gray_image, neighbors=8)
    regions = measure.regionprops(result)
    return regions

def rgb_extract(image, blocks):
    for i in blocks:
        x0, y0, x1, y1 = i.bbox
        yield image[x0:x1, y0:y1]

def extract_sequence_number(images, rgb_image):
    # plt.imshow(rgb_image)
    # plt.show()
    filtered = []
    num = 0
    for i in images:
        min_hight = i.shape[0] / 3
        max_width = i.shape[1] / 2
        regions = block_extract_t(i)
        regions.sort(key=lambda t: t.bbox[1])
        images = list(rgb_extract(rgb_image[num], regions))
        for i in images:
            height, width = np.shape(i)[:2]

            if height > min_hight and width < max_width:
                print(height, width)
                filtered.append(i)
        num = num + 1

    return filtered

def fomatImage(image):
    result = []
    width = image.shape[1]
    height = image.shape[0]
    width_4f = int(width / 4)
    result.append(image[0:height, width_4f:2 * width_4f])
    result.append(image[0:height, int(3.2 * width_4f):width])
    return result

def classify_gray(model, imgs):
    # image_shape = (32, 32, 1)
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

def saveImage(result, images):
    i = 0
    path = "img_test"
    if not os.path.exists(path):
        os.mkdir(path)
    for img in images:
        num = str(result[i])
        imgdir = path + "//" + num
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)
        imageID = str(uuid.uuid1()) + ".jpg"
        imgPath = imgdir + "//" + imageID
        print(imgPath)
        cv2.imwrite(imgPath, img)
        i = i + 1

def showImage(images):
    plt.figure()
    img_index = 1
    for img in images:
        plt.subplot(1, 10, img_index)
        plt.imshow(img)
        img_index = img_index + 1
    plt.show()

def kill_classify(image):
    images = fomatImage(image)
    showImage(images)
    #     #images_2z=images
    #
    images_2z = lap_filter1(images)
    showImage(images_2z)

    images_2z = extract_sequence_number(images_2z, images)

    showImage(images_2z)
    # images_2z=lap_filter2(images_2z)
    # showImage(images_2z)
    result = classify_gray(model, images_2z)
    # # # #     #saveImage(result,images)
    strResult = ','.join(str(s) for s in result)
    print(strResult)


# with tf.device('/gpu:0'):
model_path = 'model/model_res.h5'
print("start")
# 大坑！！！！！！,测试tensotflow里面的keras 或者单独的 keras
import keras

model = keras.models.load_model(model_path)
# model1 =load_model(model_path)
image_shape = (32, 32, 1)
print("end")
print(model.summary())

img = cv2.imread("test/test.jpg")
plt.imshow(img)
kill_classify(img)
