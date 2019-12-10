# -*- coding: utf-8 -*-
# @Time : 2019/12/10 0:37
# @Author : sxp
# @Email : 
# @File : data_type.py
# @Project : python_common

from skimage import io,img_as_float,img_as_ubyte,color
def data_type_transform(img):
    print('img type name = {}'.format(img.dtype.name))
    """1 、unit8 to float"""
    unit8_float = img_as_float(img)
    print('float_img type name = {}'.format(unit8_float.dtype.name))
    """2、float to unit8"""
    float_unit8 = img_as_ubyte(unit8_float)
    print('float_unit8 type name = {}'.format(float_unit8.dtype.name))

def color_transform(img):
    #第一种形式,rgb转灰度图
    gray = color.rgb2gray(img)
    #第二种形式,rgb转hsv
    hsv = color.convert_colorspace(img,'RGB','HSV')
    io.imshow(gray)
    io.show()
    io.imshow(hsv)
    io.show()


if __name__ == '__main__':
    path = r"/skimage/images/lena.jpg"
    img = io.imread(path)
    # data_type_transform(img)
    color_transform(img)