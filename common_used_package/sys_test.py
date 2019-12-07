# encoding: utf-8

"""
@author: sunxianpeng
@file: sys_test.py
@time: 2019/12/1 18:45
"""
import sys
import os
def sys_argv():
    #sys.argv[0]当前程序名
    # sys.argv[1]第一个参数
    # sys.argv[2]第二个参数
    # len(sys.argv)-1 参数个数(减去文件名)
    print('argv = {}'.format(sys.argv))
    print("第一个参数 = {}".format(sys.argv[1]))

def sys_path():
    print('sys path = {}'.format(sys.path))

if __name__ == '__main__':
    # sys.argv获取当前正在执行的命令行参数的参数列表(list)。
    sys_argv()
    # 返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
    sys_path()