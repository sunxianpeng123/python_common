# -*- coding: utf-8 -*-
# @Time : 2020/6/4 1:39
# @Author : sxp
# @Email : 
# @File : model.py
# @Project : flask_datatransfer
from App.extentions import db


class News(db.Model):
    id = db.Column(db.Integer,primary_key=True, autoincrement=True)
    title = db.Column(db.String(32))
    content=db.Column(db.String(256))

    def save(self):
        db.session.add(self)
        db.session.commit()
