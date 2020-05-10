# -*- coding: utf-8 -*-
# @Time : 2020/5/10 17:06
# @Author : sxp
# @Email : 
# @File : first_blue.py
# @Project : helloworld


# flask-blueprint管理路由
from flask import Blueprint

second_b = Blueprint(name='second_blue', import_name=__name__)
@second_b.route('/sb1')
def index():
    return '我是second_blue蓝图的主页'