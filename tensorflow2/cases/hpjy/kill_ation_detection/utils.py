# encoding: utf-8

"""
@author: sunxianpeng
@file: utils.py
@time: 2020/6/3 17:45
"""
import logging
import base64
import os
import uuid
import cv2
import numpy as np
from kill_ation_detection import get_this_file_name

this_file_name = get_this_file_name()
logger = logging.getLogger(this_file_name)

def saveImage(result, images):
    logger.info("save")
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

        cv2.imwrite(imgPath, img)
        i = i + 1


##############################################################
# 
##############################################################
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
        logger.info(str(r) + "得分:" + str(score))
        # print ("得分:",score,r)
        if score < 0.8:
            returnList.append(-1)
        else:
            returnList.append(r)
        r_no = r_no + 1
    return returnList


def kill_classify(images, model):
    result = []
    try:
        result = classify_gray(model, images)
        saveImage(result, images)
    except:
        logger.error("kill_classify :error image")
        result = []

    return result

##############################################################
# 
##############################################################
def imageDecode(img_b64encode,model):
    r = []
    try:
        images = []
        for img64 in img_b64encode:
            img_b64decode = base64.b64decode(img64)
            img_array = np.fromstring(img_b64decode, np.uint8)
            img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
            images.append(img)
        r = kill_classify(images,model)
    except Exception as e:
        print("error", e)
        logger.error("imageDecodee_error")
    return r
