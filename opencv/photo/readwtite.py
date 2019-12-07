# encoding: utf-8

import cv2
import matplotlib.pyplot as plt
import math
import copy
import numpy as np
# 每行三个图像，输入一个图像列表
def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

path1 = r"F:\PythonProjects\DeepLearning\image_text_recognition\data\FireShot Pro Screen Capture #017 - '星图平台-抖音、头条商业内容智能交易&管理平台' - star_toutiao_com.jpg"
path2 = r"F:\PythonProjects\python_study\opencv\detect.jpg"


img1 = cv2.imread(path1)
print("=================================")

# img2 = cv2.resize(img1,(832,832))
show_img(img1)
# print(img2.shape)
# cv2.imwrite(path2,img2)
