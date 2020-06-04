# encoding: utf-8

"""
@author: sunxianpeng
@file: utils.py
@time: 2020/6/3 11:28
"""

import logging
import cv2
import numpy as np
import json
import base64
from skimage import measure

from ctrl_rpc import get_this_file_name

this_file_name = get_this_file_name()
logger = logging.getLogger(this_file_name)


######################################################
#
######################################################
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
        logger.info("num" + str(num))

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


def lap_filter1(imgs, k_max=5):
    result = []
    for i in imgs:

        img = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        if np.sum(img) < 0.1:
            1
        else:
            kernel_lap = np.array([[0, -1, 0],
                                   [-1, k_max, -1],
                                   [0, -1, 0]])
            f1 = cv2.filter2D(img, -1, kernel_lap)
            fz = 240
            f1[f1 < fz] = 0
            f1[f1 >= fz] = 1
            result.append(f1)
    return result


def block_extract_t(gray_image):
    result = measure.label(gray_image, neighbors=8)
    regions = measure.regionprops(result)

    return regions


def kill_classify(image):
    images_2z = lap_filter1(image)
    images = extract_sequence_number(images_2z, image)
    return images


def fomatImage(image):
    result = []
    width = image.shape[1]
    height = image.shape[0]
    width_4f = int(width / 4)
    result.append(image[0:height, width_4f:2 * width_4f])
    result.append(image[0:height, int(3.2 * width_4f):width])
    return result


######################################################################
#
######################################################################


def detectkillImage(images, fibonacci_rpc, mq_info):
    logger.info("start killImage")
    jpg_as_text = []
    for img in images:
        try:
            _, buffer = cv2.imencode('.jpg', img)
            jpg64 = base64.b64encode(buffer)
        except Exception as e:
            logger.error("imencode_error")
            print("imencode", e)

        jpg_as_text.append(jpg64.decode("utf-8"))
    response = fibonacci_rpc.sendImage(jpg_as_text, "kill_info_calssify", mq_info['mq_queue_callsify_name'])
    try:
        strResult = response.decode("utf-8")
    except Exception as e:
        logger.error("detectkillImage_error")
        print("detectkillImage_error", e)
        strResult = ""
    return strResult


######################################################################
#
######################################################################
def killDetect(x, y, w, h, img, fibonacci_rpc, mq_info):
    # img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    img = img[y:h, x:w]
    cv2.imwrite("test.jpg", img)

    images = fomatImage(img)

    strResult = []
    sendJson = {};
    num_args = {'-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    if len(images) == 2:
        for i in images:
            ig = kill_classify([i])
            strResult.append(detectkillImage(ig, fibonacci_rpc, mq_info))
        alive_num_str = strResult[0].split(",")
        alive_num = ""

        for s in alive_num_str:
            if s in num_args:
                if s == '-1':
                    # alive_num='-1'
                    # break
                    1
                else:
                    alive_num = alive_num + s

        kill_num_str = strResult[1].split(",")
        kill_num = ""
        for s in kill_num_str:
            if s in num_args:
                if s == '-1':
                    # kill_num=s
                    # break
                    1
                else:
                    kill_num = kill_num + s
        if alive_num == "":
            alive_num = "-1"
        if kill_num == "":
            kill_num = "-1"
        sendJson['kill_num'] = int(kill_num)
        sendJson['alive_num'] = int(alive_num)
    return sendJson


def detectImage(image, fibonacci_rpc, mq_info):
    logger.info("start detectImage")
    _, buffer = cv2.imencode('.jpg', image)

    jpg64 = base64.b64encode(buffer)
    jpg_as_text = jpg64.decode("utf-8")

    # response = fibonacci_rpc.call(30)
    response = fibonacci_rpc.sendImage([jpg_as_text], "image_calssify", mq_info['mq_queue_yolo_name'])
    try:
        strJson = response.decode("utf-8")
        strJson = eval(strJson)
        logger.info(strJson)
        jsonResult = json.loads(str(strJson))
    except Exception as e:
        print("detectImage_error", e)
        logger.error("detectImage_error")
        jsonResult = []
    result = []
    watch = False
    for r in jsonResult:
        type = r['image_type']
        if type == "action_info":
            score = r['image_score']
            if score > 0.9:
                jsonReuslt = killDetect(r['x'], r['y'], r['w'], r['h'], image, fibonacci_rpc, mq_info)
                jsonReuslt['watch'] = "False"
                result.append(jsonReuslt)
        if type == "watch_info":
            score = r['image_score']
            if score > 0.9:
                watch = True
    if watch:
        resultWatch = []
        for r in result:
            r['watch'] = "True"
            resultWatch.append(r)
        result = resultWatch

    return result


def imageDecode(fibonacci_rpc, mq_info, img_b64encode):
    #     global fibonacci_rpc
    #     connIsClose=fibonacci_rpc.connection.is_closed
    #     print (connIsClose)
    #     if connIsClose:
    #         fibonacci_rpc = FibonacciRpcClient()
    shape = (832, 832)
    result = []
    try:
        img_b64decode = base64.b64decode(img_b64encode)
        img_array = np.fromstring(img_b64decode, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        result = detectImage(img, fibonacci_rpc, mq_info)
    except Exception as e:
        print("imageDecode", e)
        logger.error("imageDecode")
    return result
