# -*- coding: utf-8 -*-
# @Time : 2019/12/13 22:22
# @Author : sxp
# @Email : 
# @File : match_template_v1.py
# @Project : python_common

from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image

def show_image(image):
    plt.imshow(image)
    plt.show()

def save_img(path,image):
    # # opencv 通道顺序 brg，PIL 通道顺序 RGB
    print(type(image))
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.save(path)


if __name__ == '__main__':
    roi_source = r'E:/PythonProjects/深度学习数据/柱状图数据定位于识别/12.11更新/财经截图/5.jpg'
    img_dir = r'E:/PythonProjects/深度学习数据/柱状图数据定位于识别/12.11更新/财经截图'
    template_path = r'E:\PythonProjects\python_common\opencv\photo\image\template.jpg'

    template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), -1)
    template_gray = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]  # rows->h, cols->w
    print('template w = {}, h = {}'.format(w,h))

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir,img_name)
        # 读取中文图片
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray,template_gray,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        left_top = max_loc   # 左上角
        right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
        print('left top ={},right_bottom = {}'.format(left_top,right_bottom))
        cv2.rectangle(gray, left_top, right_bottom, 255, 2)  # 画出矩形位置

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(gray, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()


        break