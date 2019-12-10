# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2019/12/9 15:10
"""
import numpy as np
from skimage import measure, color
import matplotlib.pyplot as plt
import cv2

def show_image(img):
    plt.imshow(img)
    plt.show()

def showImage(images):
    plt.figure()
    img_index = 1
    for img in images:
        # print(img.shape)
        plt.subplot(1, len(images), img_index)
        plt.imshow(img)
        img_index = img_index + 1
    plt.show()

def filter2D(gray, k_max=5):
    f1 = None
    if np.sum(gray) < 0.1:
        print("black_img")
    else:
        kernel_lap = np.array([[0, -1, 0],
                               [-1, k_max, -1],
                               [0, -1, 0]])
        f1 = cv2.filter2D(gray, -1, kernel_lap)
    return f1


def get_rois(image, properties):
    list = []
    for prop in properties:
        # print("area = {},bbox = {},centroid = {}".format(prop.area, prop.bbox, prop.centroid))
        x, y, w, h = prop.bbox
        # print('x = {},y = {},w = {},h = {}'.format(x, y, w, h))
        list.append(image[x:w,y:h])

    return list

if __name__ == '__main__':
    img_path = r'F:\PythonProjects\python_common\skimage\images\number\1.jpg'
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print('img shape = {}'.format(img.shape))
    gray = filter2D(gray)
    """二值化"""
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,1)#自适应二值化
    # ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#otsu二值化

    labels = measure.label(thresh, connectivity=2,neighbors=4)  # 8连通区域标记
    print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)
    properties = measure.regionprops(labels)
    # 表示用bbox[1]作为排序的依据。即按列大小进行排序，使得到的数字顺序和图片中一致
    properties.sort(key=lambda t: t.bbox[1])
    rois = get_rois(img,properties)
    print("len(rois) = {}".format(len(rois)))

    test_rois_1 = []
    test_rois_2 = []
    test_rois_3 = []
    filtered_rois = []

    for roi in rois:
        min_hight = img.shape[0] / 5
        height, width,channel = roi.shape
        if height > min_hight and width >= 1:
            filtered_rois.append(roi)
            print('height={}, width={}'.format(height, width))
            if len(test_rois_1) < 6:
                test_rois_1.append(roi)
            else:
                if len(test_rois_2) <6:
                    test_rois_2.append(roi)
                else:
                    test_rois_3.append(roi)




    print("test_rois_1 num = {}".format(len(test_rois_1)))
    print("test_rois_2 num = {}".format(len(test_rois_2)))
    print("test_rois_3 num = {}".format(len(test_rois_3)))
    print("filtered_rois num = {}".format(len(filtered_rois)))
    showImage(test_rois_1)
    showImage(test_rois_2)
    showImage(test_rois_3)

