# encoding: utf-8

"""
@author: sunxianpeng
@file: 7_imgContours.py
@time: 2020/8/12 14:18
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def showImg(img):
    """显示单个图像"""
    plt.imshow(img)
    plt.show()


def showImgs(imgs):
    """传入图片的List，显示多个图片，"""
    plt.figure()
    img_index = 1
    for img in imgs:
        # print(img.shape)
        plt.subplot(1, len(imgs), img_index)
        plt.imshow(img)
        img_index = img_index + 1
    plt.show()


# ==============================================================================
# 读取图片
# ==============================================================================
def readImg(image_path, path_has_chinese=True):
    """
    读取图片
    :param image_path:图片路径
    :param path_has_chinese: 路径中是否有中文，默认有中文
    :return:
    """
    if not path_has_chinese:
        img = cv2.imread(image_path)
    else:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    # print(isinstance(img, type(None)))
    # 判断img是否为None
    if isinstance(img, type(None)) == True:  raise Exception("File Not Found!!")
    return img


# ==============================================================================
# 1、轮廓定义
#       轮廓可以简单认为成连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度。轮廓在形状分析和物体的检测和识别中很有用。
# 2、注意
#       为了准确，要使用二值化图像。需要进行阀值化处理或者Canny边界检测。
#       查找轮廓的函数会修改原始图像。如果之后想继续使用原始图像，应该将原始图像储存到其他变量中。
#       在OpenCV中，查找轮廓就像在黑色背景中超白色物体。你应该记住，要找的物体应该是白色而背景应该是黑色
# 3、如何在一个二值图像中查找轮廓。
#       函数 cv2.findContours() ：
#           有三个参数，第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法。
#           返回值有三个，第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构。轮廓（第二个返回值）是一个Python列表，其中储存这图像中所有轮廓。每一个轮廓都是一个Numpy数组，包含对象边界点（x，y）的坐标。
#                   轮廓检索模式 	        含义
#                   cv2.RETR_EXTERNAL 	    只检测外轮廓
#                   cv2.RETR_LIST 	        提取所有轮廓并将其放入列表,不建立等级关系
#                   cv2.RETR_CCOMP 	        建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层
#                   cv2.RETR_TREE 	        建立一个等级树结构的轮廓

#                   轮廓逼近方法 	                                        含义
#                   cv2.CHAIN_APPROX_NONE 	                                存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
#                   cv2.CHAIN_APPROX_SIMPLE 	                            压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
#                   cv2.CHAIN_APPROX_TC89_L1 或 cv2.CHAIN_APPROX_TC89_KCOS 	应用Teh-Chin链近似算法
# 4、怎样绘制轮廓
# （1）要绘制图像中的所有轮廓
#       cv.drawContours（img，contours，-1，（0,255,0），3）
# （2）要绘制单个轮廓，比如第4个轮廓
#       cv.drawContours（img，contours，3，（0,255,0），3）
# （3）但大多数情况下，绘制第4个轮廓，以下方法将非常有用
#     cnt = contours[4]
#     cv.drawContours（img，cnt，0，（0,255,0），3）
# 5、opencv 版本问题，造成 findContours 方法返回参数个数不同
#           E:\ProgrammeSoftware\Anaconda\python.exe E:/PythonProjects/python_study/opencv/pyramid.py
#           Traceback (most recent call last):
#               File "E:/PythonProjects/python_study/opencv/pyramid.py", line 18, in <module>
#               images,contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#           ValueError: not enough values to unpack (expected 3, got 2)
#           Process finished with exit code 1
#  出现错误原因：
#       如果用的是 openCV 4.0版本，findContours返回的是两个参数，旧版的返回的则是三个参数
# 解决方法：
#       移除第一个参数赋值，将
#       image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#       改为
#       cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# ==============================================================================
def imgFindContours(binary):
    """
    :param binary:
    :return:
    """
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_cp = img.copy()
    r = cv2.drawContours(img_cp, contours, -1, (255, 0, 0), 3)
    showImgs([img, r])


if __name__ == '__main__':
    read_path_lena = os.path.join(os.path.abspath("/PythonProjects\python_common\opencv\data\image\exercise"),
                                  "lena.jpg")
    print("read_path_lena = {}".format(read_path_lena))
    img = readImg(read_path_lena, False)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(img_gray, 127, 255, 0)
    print("img1 shape = {}".format(img.shape))
    imgFindContours(binary)
