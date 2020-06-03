# encoding: utf-8

"""
@author: sunxianpeng
@file: __init__.py.py
@time: 2020/6/2 19:06
"""

import tensorflow as tf
import sys
import logging
import time
import os
def get_this_file_name():
    """"""
    # print('sys.argv[0]  = {}'.format(sys.argv[0] ))
    this_py_path = sys.argv[0]  # 第0个就是这个python文件本身的路径（全路径）
    this_py_name = os.path.basename(this_py_path)  # 当前文件名名称
    # print(os.path.basename(__file__))# 当前文件名名称)
    this_file_name = this_py_name.split('.')[0]
    return this_file_name

def get_logger(log_file, this_file_name):
    """日志设置"""
    # 第一步，创建一个logger
    logger = logging.getLogger(this_file_name)
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    fh = logging.FileHandler(log_file, mode='w')
    ch = logging.StreamHandler()
    fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
    ch.setLevel(logging.INFO)
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger



def check_device():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("{} Physical GPUs, {} Logical GPUs".format(len(gpus), len(logical_gpus)))
    # except RuntimeError as e:
    except Exception as e:
        # Memory growth must be set before GPUs have been initialized
        print("error")
