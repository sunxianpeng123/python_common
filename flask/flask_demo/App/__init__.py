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

from App.views.first_blue import first_b
from App.views.ordinary_views import init_route
from App.views.second_blue import second_b


def create_app():
    app = Flask(__name__)
    ###############################
    # 操作数据库 models orm配置
    ###############################
    # db_uri格式:数据库:驱动//用户名:密码@主机:端口/数据库名字
    # db_uri = "mysql+pymysql://username:password@server:port/db"
    # app.config['SQLALCHEMY_DATABASE_URI'] = mysql_uri
    db_uri = "sqlite:///sqlite.db"
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    init_extentions(app)
    ###############################
    # 路由配置
    ###############################
    "普通方式管理"
    init_route(app)
    "blueprint方式管理"
    init_blue_views(app)
    # app.add_url_rule('/', endpoint='index')
    return app

def init_blue_views(app):
    app.register_blueprint(first_b)
    app.register_blueprint(second_b)
