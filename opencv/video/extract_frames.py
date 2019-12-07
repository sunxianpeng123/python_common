# encoding: utf-8

"""
@author: sunxianpeng
@file: extract_frames.py
@time: 2019/11/29 17:31
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

def extract_frame_and_save(video_path,extrach_folder,extrach_frequency,file_mid_path):
    count = 1
    cap = cv2.VideoCapture(video_path)
    # 判断是否正常打开
    if cap.isOpened():
        ret,frame = cap.read()
    else:
        ret = False
    # 循环读取视频帧
    while ret:
        ret ,frame = cap.read()
        # 每隔extrach_frequency帧进行存储操作
        if  count % extrach_frequency == 0:
            # 存储为图像
            file_path = extrach_folder + '/' + file_mid_path + '_' + str(count) + '.jpg'
            print(file_path)
            cv2.imwrite(file_path,frame)
        count += 1
        cv2.waitKey(1)
    cap.release()

def get_video_info_list():
    video_info_list = []
    video_path_1 = r'F:\PythonProjects\python_study\opencv\video\余小c动作未识别taskidA5B8C090929308AEEDEB18FADC61D0035DE77EAE\3min27s到3min41s.mp4'  # 视频地址
    video_path_2 = r'F:\PythonProjects\python_study\opencv\video\余小c动作未识别taskidA5B8C090929308AEEDEB18FADC61D0035DE77EAE\5min46到6min30.mp4'  # 视频地址
    video_path_3 = r'F:\PythonProjects\python_study\opencv\video\余小c动作未识别taskidA5B8C090929308AEEDEB18FADC61D0035DE77EAE\50s.mp4'  # 视频地址
    file_path_1 = '3min27s_3min41s'
    file_path_2 = '5min46_6min30'
    file_path_3 = '50s'

    video_info_list.append((video_path_1, file_path_1))
    video_info_list.append((video_path_2, file_path_2))
    video_info_list.append((video_path_3, file_path_3))
    return video_info_list



if __name__ == '__main__':
    # 全局变量
    video_info_list = get_video_info_list()
    extrach_folder = r'F:\PythonProjects\python_study\opencv\video\extract_folder'  # 存放帧图片的位置
    extrach_frequency = 50 # 帧提取频率

    for tup in video_info_list:
        video_path = tup[0]
        file_mid_path = tup[1]
        print('video_path = {}'.format(video_path))
        extract_frame_and_save(video_path,extrach_folder,extrach_frequency,file_mid_path)


