# encoding: utf-8

"""
@author: sunxianpeng
@file: detect_objects_in_video.py
@time: 2020/6/25 19:33
"""
import tensorflow as tf
import cv2
from configuration import test_video_dir, temp_frame_dir, CATEGORY_NUM, save_model_dir
from test_on_single_image import single_image_inference
from yolo.yolo_v3 import YOLOV3


def frame_detection(frame, model):
    cv2.imwrite(filename=temp_frame_dir, img=frame)
    frame = single_image_inference(image_dir=temp_frame_dir, model=model)
    return frame