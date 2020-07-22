# encoding: utf-8

"""
@author: sunxianpeng
@file: connectedComponentswithstats.py
@time: 2019/12/10 10:59
"""
import os

import numpy as np
import scipy.ndimage as ndi
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


def get_background(stats):
    """获取background的行号"""
    background = None
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
        else:
            continue
    return background


def get_rois(image, stats):
    list = []
    for stat in stats:
        x, y, w, h, area = stat
        # print('x = {},y = {},w = {},h = {}'.format(x, y, w, h))
        list.append(image[y:y + h, x:x + w])
    return list


def morphology(thresh, type='dilate'):
    """形态学转换"""
    result = None
    if type == 'dilate':
        # 膨胀
        size = 2
        kernal = np.ones((size, size), np.uint8)
        result = cv2.dilate(thresh, kernal, iterations=1)
    if type == 'erode':
        size = 2
        kernel = np.ones((size, size), np.uint8)
        result = cv2.erode(thresh, kernel, iterations=1)
    return result


if __name__ == '__main__':
    # stats 是bounding box的信息，N*5的矩阵，行对应每个label，五列分别为（x,y,w,h,area）
    # centroids是每个域的质心坐标

    img_path = os.path.abspath("./image/number/8.jpg")
    # img_path = r'F:\PythonProjects\python_common\opencv\photo\image\number\8.jpg'
    img = cv2.imread(img_path)
    print('img shape = {}'.format(img.shape))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = filter2D(gray)
    """二值化"""
    ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,1)#自适应二值化
    # ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#otsu二值化
    """形态学转换"""
    # show_image(thresh)
    # thresh = morphology(thresh)
    # show_image(thresh)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

    background = get_background(stats)
    stats = np.delete(stats, background, axis=0)
    # print(stats)

    stats = stats[stats[:, 0].argsort()]  # 按照第一列的值，对行进行排序
    print('label shape = {},stats.shape = {},centroids.shape = {}'.format(labels.shape, stats.shape, centroids.shape))
    # print(stats)
    print(centroids)
    rois = get_rois(img, stats)

    filtered_rois = []
    roi_width_10 = []
    for roi in rois:
        min_hight = img.shape[0] / 6
        x, y, channel_1 = img.shape
        height, width, channel_2 = roi.shape
        percent = width / height
        if height > min_hight and width >= 1 and height != x and width != y:
            print('height={}, width={},width / height = {}'.format(height, width, width / height))
            if width > 10 and height > 7 and percent > 1.2:  # 符合条件的图片进行再次切分
                print('###########################height={}, width={},width / height = {}'.format(height, width,
                                                                                                  width / height))
                # roi_width_10.append(roi)
                if width >= 11 and width <= 14:  # 再次切分成两个
                    mid = width // 2
                    filtered_rois.append(roi[:, :mid])
                    filtered_rois.append(roi[:, mid:])
                if width >= 15 and width <= 37:  # 再次切分成三个
                    mid_1 = width // 3
                    mid_2 = mid_1 * 2
                    filtered_rois.append(roi[:, :mid_1 + 1])
                    filtered_rois.append(roi[:, mid_1 + 1:mid_2 + 1])
                    filtered_rois.append(roi[:, mid_2 + 1:])
            else:
                filtered_rois.append(roi)
    print("filtered_rois num = {}".format(len(filtered_rois)))
    print("roi_width_10 num = {}".format(len(roi_width_10)))
    showImage(filtered_rois)
    showImage(roi_width_10)
