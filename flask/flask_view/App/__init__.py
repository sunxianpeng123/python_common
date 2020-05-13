# -*- coding: utf-8 -*-
# @Time : 2020/5/10 21:16
# @Author : sxp
# @Email : 
# @File : __init__.py.py
# @Project : helloworld


from flask import Flask

# blueprint方式管理
from flask_sqlalchemy import SQLAlchemy

from App.extentions import  init_extentions
from App.settings import envs
from App.views import init_view


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

