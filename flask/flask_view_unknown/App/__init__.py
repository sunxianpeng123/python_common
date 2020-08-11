# -*- coding: utf-8 -*-
# @Time : 2020/5/10 21:16
# @Author : sxp
# @Email : 
# @File : __init__.py.py
# @Project : helloworld


from flask import Flask

# blueprint方式管理

from App.extentions import  init_extentions
from App.settings import envs
from App.views.template_view.bootstrap_view import bootstap_view_blue
from App.views.template_view.demo import template_view_blue
from App.views.view_view.base import base_blue
from App.views.view_view.cookies import cookies_blue
from App.views.view_view.redirection import redirection_blue
from App.views.view_view.request import request_blue
from App.views.view_view.request_method import req_method_blue
from App.views.view_view.response import response_blue

def create_app(env):
    app = Flask(__name__)
    # 加载配置
    app.config.from_object(envs.get(env))
    # 初始化第三方扩展库
    init_extentions(app)
    # 初始化路由
    init_view(app)
    # app.add_url_rule('/', endpoint='index')
    return app

def init_view(app):
    app.register_blueprint(base_blue)
    app.register_blueprint(req_method_blue)
    app.register_blueprint(redirection_blue)
    app.register_blueprint(request_blue)
    app.register_blueprint(response_blue)
    app.register_blueprint(cookies_blue)
#     template
    app.register_blueprint(template_view_blue)
    app.register_blueprint(bootstap_view_blue)
