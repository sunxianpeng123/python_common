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
from App.views.insert.ordinary import ordinary_insert_blue
from App.views.delete.ordinary import ordinary_delete_blue
from App.views.select.condition import condition_select_blue
from App.views.select.ordinary import ordinary_select_blue
from App.views.update.ordinary import ordinary_update_blue


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
    # 增
    app.register_blueprint(ordinary_insert_blue)
    # 删
    app.register_blueprint(ordinary_delete_blue)
    # 改
    app.register_blueprint(ordinary_update_blue)
    # 查
    app.register_blueprint(ordinary_select_blue)
    app.register_blueprint(condition_select_blue)

