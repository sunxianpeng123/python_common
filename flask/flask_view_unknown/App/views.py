# # -*- coding: utf-8 -*-
# # @Time : 2020/5/11 0:42
# # @Author : sxp
# # @Email :
# # @File : __init__.py.py
# # @Project : helloworld
# from flask import Blueprint, redirect, url_for
#
# blue = Blueprint('blue', __name__)
# def init_view(app):
#     app.register_blueprint(blue)
#
# # http://127.0.0.1:5000/
# @blue.route('/')
# def index():
#     return  'index'
#
# #######################################################
# # 1、单个路由
# #######################################################
# # 接受输入参数,并指定类型
# # int,float 直接写
# # string 默认,他是以 / 作为结尾的
# # path 会将getpath之后的所有都当做参数
# # uuid 只接受uuid格式字符串，唯一码
# # any  可以同时指定多种路径，进行限定,类似于穷举,只能写存在于any中的值
# # http://127.0.0.1:5000/user/19/
# @blue.route('/user/<int:id>/')
# def users(id):
#     print('id = {}, type = {}'.format(id, type(id)))
#     return 'user = {}'.format(id)
#
# # http://127.0.0.1:5000/getinfo1/aa/
# @blue.route('/getinfo1/<string:token>/')
# def get_info(token):
#     print('token = {}, type = {}'.format(token, type(token)))
#     return 'get token = {}'.format(token)
#
# # http://127.0.0.1:5000/getpath/dir1/dir2
# @blue.route('/getpath/<path:address>/')
# def get_path(address):
#     print('path = {}, type = {}'.format(address, type(address)))
#     return 'get path = {}'.format(address)
#
# # http://127.0.0.1:5000/getuuid/1e9aec82-34f4-4743-921f-01cd98682077
# @blue.route('/getuuid/<uuid:uu>/')
# def get_uuid(uu):
#     print('uuid = {}, type = {}'.format(uu, type(uu)))
#     return 'get uuid = {}'.format(uu)
#
# # http://127.0.0.1:5000/getany/
# @blue.route('/getany/<any(a,b):an>/')
# def get_any(an):
#     print('any = {}, type = {}'.format(an, type(an)))
#     return 'get any = {}'.format(an)
#
# #######################################################
# #2、多个路由
# #######################################################
# # 可以同时支持 string 和 int 类型
# # http://127.0.0.1:5000/getinfo2/aa/
# @blue.route('/getinfo2/<string:token>/')
# @blue.route('/gettoken2/<int:token>/')
# def get_info2(token):
#     print('token = {}, type = {}'.format(token, type(token)))
#     return 'get token = {}'.format(token)
#
# #######################################################
# #3、请求方法 ，默认支持：Get、Head、Options，其余请求方法需要手动注册
# #######################################################
# # http://127.0.0.1:5000/method
# @blue.route('/method/', methods=['get', 'post', 'delete'])
# def method():
#     return 'request method'
#
# #######################################################
# #4、重定向，反向解析
# #######################################################
# # http://127.0.0.1:5000/redirect1
# @blue.route('/redirect1/')
# def red_1():
#     # /method/指的是路由的函数名字
#     return redirect('/method/')
#
# # http://127.0.0.1:5000/redirect2
# @blue.route('/redirect2/')
# def red_2():
#     # 反向解析
#     # blue.index 用了蓝图就要写成这种格式
#     # 没有用蓝图，则只需要写要重定向到的视图的函数名字即可
#     return redirect(url_for('blue.index'))
#
# # http://127.0.0.1:5000/redirect3/
# @blue.route('/redirect3/')
# def red_3():
#     return redirect(url_for('blue.get_any', an='a'))
#
# #######################################################
# #4、重定向，反向解析
# #######################################################