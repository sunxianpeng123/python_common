# -*- coding: utf-8 -*-
# @Time : 2020/7/18 18:38
# @Author : sxp
# @Email : 
# @File : convert.py
# @Project : yolov3


import os
import re
import shutil
import cv2
import random


def extract_content(content_):
    # 注意，一开始用的第一种，结果只有一行的情况没有被提取出来，要去掉后面的\n，谨记
    # content_extract = re.findall('(.*?) (.*?) (.*?) (.*?) (.*?)\n', content)
    content_extract_ = re.findall('(\d+.?\d*) (\d+.?\d*) (\d+.?\d*) (\d+.?\d*) (\d+.?\d*)', content_)

    return content_extract_


if __name__ == '__main__':

    # 记得路径尾部加“/”，不然调用join方法是它会用“\”替代，那样不好，容易造成转义字符问题。
    # ../表示上一层路径
    # 该路径会写入train.txt和test.txt，tensorflow-yolov3训练时使用，相对于train.py的路径
    source_img_path_related_to_train_py = './data/dataset/imageset/'

    # 转换格式时所使用的路径，相对于convert.py的路径
    source_img_path = './imageset/'

    # 用labelImg标注的yolo格式的目标信息文件路径。与训练图集中的图像一一对应，名称相同，类型为txt文档。
    source_txt_path = './yolotxt/'

    # yolo格式转换为tensorflow2-yolo3格式后的文件存储路径
    target_txt_path = './'

    # train.txt:记录训练用图像路径以及相应图像中目标相关信息
    # test.txt:记录测试用图像路径以及相应图像中目标相关信息
    # 打开上述两个文件准备写入
    train_file = open(target_txt_path + 'train.txt', 'w', encoding='utf-8')
    test_file = open(target_txt_path + 'test.txt', 'w', encoding='utf-8')

    # 创建写入内容字符串变量
    train_file_content = ''
    test_file_content = ''

    # 该参数与实际训练类别数量相同
    MAX_ID = 1

    # 读取source_txt_path路径下所有文件（包括子文件夹下文件）
    filenames = os.listdir(source_txt_path)

    # 打开labelImg标注的yolo格式的文件，提取其中数字并将内容重构后写入新文件
    for filename in filenames:

        # 打开文件：
        with open(os.path.join(source_txt_path, filename), 'r', encoding='utf-8') as f:

            # 读取文件内容
            content = f.read()

            # 提取数据
            content_extract = extract_content(content)

            # 获取当前图片分辨率信息（这样不论图片尺寸多少都能成功转换）（re.findall()返回的是列表，需要将它转换成字符串）
            # 读取图片
            tmpfilename = '{}{}.jpg'.format(source_img_path, ''.join(re.findall('(.*?).txt', filename)))
            print(tmpfilename)
            img = cv2.imread(tmpfilename)

            # 获取图片分辨率
            img_width = img.shape[1]
            img_height = img.shape[0]

            # 创建单行写入字符串的路径头字符串：图片路径+图片名称
            path_str = source_img_path_related_to_train_py + os.path.splitext(filename)[0] + '.jpg'
            print(path_str)

            # 创建单行写入字符串的目标坐标字符串
            obj_strs = ''

            # 将数据格式从相对坐标转换成绝对坐标
            for obj_str in content_extract:

                # 将元组字符串转换成列表数字
                object_evar = list(map(eval, obj_str))

                # 映射变量
                class_id = object_evar[0]

                x, y = object_evar[1] * img_width, object_evar[2] * img_height

                w, h = object_evar[3] * img_width, object_evar[4] * img_height

                # 判断数据是否超出限制（数据清洗）（包括清洗超限坐标和错误class_id）
                if class_id >= MAX_ID \
                        or round(x - w / 2) < 0 \
                        or round(x + w / 2) > img_width \
                        or round(x - w / 2) >= round(x + w / 2) \
                        or round(y - h / 2) < 0 \
                        or round(y + h / 2) > img_height \
                        or round(y - h / 2) >= round(y + h / 2):
                    print('错误标注：')
                    print(filename)
                    print(object_evar)
                    print('[{}, {}, {}, {}, {}]'.format(round(x - w / 2), round(y - h / 2), round(x + w / 2),
                                                        round(y + h / 2), class_id))
                    continue

                # 将映射变量格式化后加入到obj_strs中：
                obj_strs += ' {},{},{},{},{}'.format(round(x - w / 2), round(y - h / 2), round(x + w / 2),
                                                     round(y + h / 2), class_id)

            # 拆分训练集和测试集
            # 训练集占比
            train_scale = 0.75

            # 设置随机概率
            proba = random.random()

            # 如果该张图片经过数据清洗后没有目标，则跳过，不将其加入到train.txt和test.txt文件中
            if obj_strs == '':
                print('空文件：{}'.format(filename))
                print('content：{}'.format(content))
                cv2.imwrite('null_img\\{}.jpg'.format(''.join(re.findall('(.*?).txt', filename))), img)
                print('将图片拷贝到“空文件”文件夹')
                continue
            else:
                write_strs = path_str + obj_strs
            print(write_strs)

            # 判断该写入哪个文件
            if proba < train_scale:
                train_file_content += write_strs + '\n'
            else:
                test_file_content += write_strs + '\n'

    # 将两个即将写入的内容去除首位的无效字符（如空格，换行符，制表符，回车符）
    train_file_content = train_file_content.strip()
    test_file_content = test_file_content.strip()

    train_file.write(train_file_content)
    test_file.write(test_file_content)

    train_file.close()
    test_file.close()
