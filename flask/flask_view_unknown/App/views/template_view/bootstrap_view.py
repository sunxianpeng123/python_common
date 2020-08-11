# -*- coding: utf-8 -*-
# @Time : 2020/5/20 0:34
# @Author : sxp
# @Email : 
# @File : bootstrap_view.py
# @Project : flask_view

from flask import Blueprint, render_template

bootstap_view_blue = Blueprint('bootstap_view_blue', __name__)
##########################################################
# 简洁、直观、强悍的前端开发框架，让web开发更迅速、简单
# https://www.bootcss.com/
# https://v3.bootcss.com/   获取模板
##########################################################
# http://127.0.0.1:5000/boorstrapindex/
@bootstap_view_blue.route('/boorstrapindex/', methods=['get', 'post', 'delete'])
def get_index():
    return render_template('bootstrap_test/index.html')




