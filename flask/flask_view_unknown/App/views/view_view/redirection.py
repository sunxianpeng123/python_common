# encoding: utf-8

"""
@author: sunxianpeng
@file: redirection.py
@time: 2020/5/16 18:49
"""
from flask import Blueprint, redirect, url_for

redirection_blue = Blueprint('redirection_blue', __name__)
#######################################################
#4、重定向，反向解析
#######################################################
# http://127.0.0.1:5000/redirect1/
@redirection_blue.route('/redirect1/')
def red_1():
    # /method/指的是路由的函数名字
    return redirect('/method/')

# http://127.0.0.1:5000/redirect2/
@redirection_blue.route('/redirect2/')
def red_2():
    # 反向解析
    # blue.index 用了蓝图就要写成这种格式
    # 没有用蓝图，则只需要写要重定向到的视图的函数名字即可
    return redirect(url_for('base_blue.index'))

# http://127.0.0.1:5000/redirect3/
@redirection_blue.route('/redirect3/')
def red_3():
    return redirect(url_for('base_blue.get_any', an='a'))

