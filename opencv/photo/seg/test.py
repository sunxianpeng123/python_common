# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2020/5/11 18:55
"""
import os

import PIL
import numpy
import matplotlib.pyplot as plt


def OnClick(event):
    # 获取当鼠标"按下"的时候，鼠标的位置
    global Coords1x, Coords1y
    if event.button == 1:
        try:
            Coords1x = int(event.xdata)
            Coords1y = int(event.ydata)
        except:
            Coords1x = event.xdata
            Coords1y = event.ydata
        print("#### 左上角坐标 ：", Coords1x, Coords1y)


def OnMouseMotion(event):
    # 获取当鼠标"移动"的时候，鼠标的位置
    global Coords2x, Coords2y
    if event.button == 3:
        try:
            Coords2x = int(event.xdata)
            Coords2y = int(event.ydata)
        except:
            Coords2x = event.xdata
            Coords2y = event.ydata
        print("#### 右下角坐标 ：", Coords2x, Coords2x)

def get_screen_image():
    img_path = '../data\grab_cut.jpg'
    return numpy.array(PIL.Image.open(img_path))

figure = plt.figure()
axes_image = plt.imshow(get_screen_image(), animated = True)
figure.canvas.mpl_connect('button_press_event', OnClick)
figure.canvas.mpl_connect('button_release_event', OnMouseMotion)

plt.show()