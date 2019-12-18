# -*- coding: utf-8 -*-
# @Time : 2019/12/19 0:11
# @Author : sxp
# @Email : 
# @File : paramiko_test.py
# @Project : python_common
import paramiko



"""paramiko是一个用于做远程控制的模块，使用该模块可以对远程服务器进行命令或文件操作，值得一说的是，fabric和ansible内部的远程管理就是使用的paramiko来现实。"""
# 1、下载安装
# pycrypto，由于 paramiko 模块内部依赖pycrypto，所以先下载安装pycrypto
# pip3 install pycrypto
# pip3 install paramiko
"""参考链接"""
# https://blog.csdn.net/forever_wen/article/details/82556154

if __name__ == '__main__':
    ip = '192.168.122.3'
    port = 22
    user = 'root'
    password = '123456'
