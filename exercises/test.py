# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_indoor.py
@time: 2019/11/14 14:12
"""
import torch
import torchvision
from torchvision import datasets,transforms

import skimage.io as io
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    s = [(b'hist_title', 0.9737505316734314, (456, 1110, 793, 1137))]
    for one in s:
        image_type = one[0]
        score = one[1]
        x,y,w,h = one[2]
        print(x,y,w,h)



