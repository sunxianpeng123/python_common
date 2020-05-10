# -*- coding: utf-8 -*-
# @Time : 2020/5/10 16:11
# @Author : sxp
# @Email : 
# @File : models.py
# @Project : helloworld
from flask_sqlalchemy import SQLAlchemy

from App.extentions import db


class User(db.Model):
    # 对应数据库中的 user 表
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(16))
    pass



if __name__ == '__main__':
    pass