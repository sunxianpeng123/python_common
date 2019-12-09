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

def show_img(img):
    io.imshow(img)
    io.show()

def save_img(path,img):
    io.imsave(path,img)#也起到了转换格式的作用

if __name__ == '__main__':
    path = r"/skimage/images/lena.jpg"
    save_path = r'/skimage/save.jpg'
    img = read_rgb_or_gray(path,is_self_img=True,type='rgb')

    show_img(img)
    save_img(save_path,img)
