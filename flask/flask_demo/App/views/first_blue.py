# -*- coding: utf-8 -*-
# @Time : 2020/5/10 17:06
# @Author : sxp
# @Email : 
# @File : first_blue.py
# @Project : helloworld


# flask-blueprint管理路由
from flask import Blueprint, render_template

from App.models import db

first_b = Blueprint('first_blue',__name__)

@first_b.route('/fb1')
def index():
    # return '我是first_blue蓝图的主页'
    # 传递 msg 到html，变量名字就是 msg
    return render_template('index.html', msg="这天气适合睡觉！")

@first_b.route('/createdb/')
def createdb():
    db.create_all()
    return '创建db成功'