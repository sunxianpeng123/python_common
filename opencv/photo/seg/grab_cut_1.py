# encoding: utf-8

"""
@author: sunxianpeng
@file: grab_Cut.py
@time: 2020/5/11 18:17
"""
import cv2
import numpy as np
from matplotlib import pyplot as  plt

"""
运用GrabCut轻松玩转抠图
GrabCut该算法利用了图像中的纹理（颜色）信息和边界（反差）信息，只要小量的用户交互操作即可得到比较好的分割效果
首先用矩形将要选择的前景区域选定，其中前景区域应该完全包含在矩形框当中。
然后算法进行迭代式分割，知道达到效果最佳。但是有时分割结果不好，例如前景当成背景，背景当成前景。测试需要用户修改。
用户只需要在非前景区域用鼠标划一下即可。如文档中的图片，运动员和足球被蓝色矩形保卫，其中有数个用白色标记修改的，表示前景区域，黑色表示背景区域。
函数原型：
    grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
        img - 输入图像
        mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
        rect - 包含前景的矩形，格式为 (x,y,w,h)
        bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
        iterCount - 算法的迭代次数
        mode - cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
"""
class Main():
    def __init__(self):
        pass

def show_img(img):
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    img_path = '../data\grab_cut.jpg'
    img = cv2.imread(img_path)
    mask = np.zeros(img.shape[:2], np.uint8)

    # zeros(shape, dtype=float, order='C')，参数shape代表形状，(1,65)代表1行65列的数组，dtype:数据类型，可选参数，默认numpy.float64
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, img.shape[1], img.shape[0])
    # 函数原型：grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None)
    # img - 输入图像
    # mask-掩模图像，用来确定那些区域是背景，前景，可能是前景/背景等。可以设置为：cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，或者直接输入 0,1,2,3 也行。
    # rect - 包含前景的矩形，格式为 (x,y,w,h)
    # bdgModel, fgdModel - 算法内部使用的数组. 你只需要创建两个大小为 (1,65)，数据类型为 np.float64 的数组。
    # iterCount - 算法的迭代次数
    # mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。

    x = 1 + 8
    y = 1 + 8
    w = img.shape[1] - 55
    h = img.shape[0] - 25
    # a = img[y:h, x:w]
    # show_img(a)
    # exit(0)
    # rect = (x, y, w, h)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    # np.where 函数是三元表达式 x if condition else y的矢量化版本
    # result = np.where(cond,xarr,yarr)
    # 当符合条件时是x，不符合是y，常用于根据一个数组产生另一个新的数组。
    # | 是逻辑运算符or的另一种表现形式
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # mask2[:, :, np.newaxis] 增加维度
    img = img * mask2[:, :, np.newaxis]

    # 显示图片
    plt.subplot(121), plt.imshow(img)
    plt.title("grabcut"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
    plt.title("original"), plt.xticks([]), plt.yticks([])
    plt.show()

