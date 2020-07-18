# -*- coding: utf-8 -*-
# @Time : 2020/5/11 1:27
# @Author : sxp
# @Email : 
# @File : extentions.py
# @Project : helloworld

# 放置第三方扩展库
from flask_bootstrap import Bootstrap
from flask_caching import Cache

from flask_debugtoolbar import DebugToolbarExtension
from flask_migrate import Migrate
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy

migrate = Migrate()
# 数据库模型信息初始化
db = SQLAlchemy()

def init_extentions(app):
    db.init_app(app=app)
    migrate.init_app(app, db)
    Session(app)
    Bootstrap(app)
    DebugToolbarExtension(app)