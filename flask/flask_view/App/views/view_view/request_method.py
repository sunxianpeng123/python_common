# encoding: utf-8

"""
@author: sunxianpeng
@file: req_method.py
@time: 2020/5/16 18:48
"""
from flask import Blueprint

req_method_blue = Blueprint('req_method_blue', __name__)

#######################################################
#3、请求方法 ，默认支持：Get、Head、Options，其余请求方法需要手动注册
# """ 需要搞清楚每种请求方法的原理"""
#######################################################
# http://127.0.0.1:5000/method/
@req_method_blue.route('/method/', methods=['get', 'post', 'delete'])
def method():
    return 'request method'
