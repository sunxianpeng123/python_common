# -*- coding: utf-8 -*-
# @Time : 2020/5/10 16:11
# @Author : sxp
# @Email : 
# @File : models.py
# @Project : helloworld
from flask_sqlalchemy import SQLAlchemy

from App.extentions import db


class User(db.Model):
    # 对应数据库中的 User 表
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(16))

    def save(self):
        db.session.add(self)
        db.session.commit()

class Student(db.Model):
    # 对应数据库中的 Student 表
    id = db.Column(db.Integer, primary_key=True)
    s_name = db.Column(db.String(16))
    s_password = db.Column(db.String(256))

    def save(self):
        db.session.add(self)
        db.session.commit()