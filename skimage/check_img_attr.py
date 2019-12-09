# -*- coding: utf-8 -*-
# @Time : 2019/12/9 23:56
# @Author : sxp
# @Email : 
# @File : read_show_save.py
# @Project : python_common

from skimage import io,data,data_dir

def read_rgb_or_gray(path=None,is_self_img=True,type='rgb'):
    img = None
    if is_self_img:
        if type == 'rgb':
            img = io.imread(path)#读取单张彩色rgb图片
        if type == 'gray':
            img = io.imread(path,as_gray=True)#读取单张灰度图片
    else:
        print('data_dir = {}'.format(data_dir))#图片存放在skimage的安装目录下面，路径名称为data_dir,
        img = data.camera()#程序自带图片
    return img


if __name__ == '__main__':
    path = r"/skimage/images/lena.jpg"
    save_path = r'/skimage/save.jpg'
    img = read_rgb_or_gray(path,is_self_img=True,type='rgb')
    print(type(img))  # 显示类型
    print('img.shape = {}'.format(img.shape))  # 显示尺寸
    print('图片宽度 = {}'.format(img.shape[0]))  #
    print('图片高度 = {}'.format(img.shape[1]))  #
    print('图片通道数 = {}'.format(img.shape[2]))  #
    print('显示总像素个数 = {}'.format(img.size))  #
    print('最大像素值 = {}'.format(img.max()))  #
    print('最小像素值 = {}'.format(img.min()))  #
    print('像素平均值 = {}'.format(img.mean()))  #
