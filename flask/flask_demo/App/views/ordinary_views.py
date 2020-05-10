# -*- coding: utf-8 -*-
# @Time : 2020/5/10 16:12
# @Author : sxp
# @Email : 
# @File : ordinary_views.py
# @Project : helloworld
#

from flask import Blueprint

# 普通方式管理路由
def init_route(app):
    @app.route('/')
    def hello_world():
        return 'Hello World!'

    @app.route('/t1')
    def  hello():
        return 'hello 个毛线！'

    @app.route('/t2')
    def hi():
        return 'hi 我是hi！'

