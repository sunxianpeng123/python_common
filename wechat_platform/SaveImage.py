# encoding: utf-8

"""
@author: sunxianpeng
@file: SaveImage.py
@time: 2020/5/1 22:52
"""


class SaveImage():
    def __init__(self):
        pass

    def LoadImageFromURL(self, imgname, image_bytes):
        with open(imgname, 'wb') as f:
            f.write(image_bytes)  # 将内容写入图片
        f.close()

if __name__ == '__main__':
    pass