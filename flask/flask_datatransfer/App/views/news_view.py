# encoding: utf-8

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""
# encoding: utf-8
from App.extentions import db
from App.models.model import News

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""

from flask import Blueprint, render_template

blue = Blueprint('blue', __name__)
# http://127.0.0.1:5000/
@blue.route('/')
def index():
    return 'index'

# http://127.0.0.1:5000/createdb/
@blue.route('/createdb/')
def createdb():
    db.create_all()
    return '创建db成功'

# http://127.0.0.1:5000/dropdb/
@blue.route('/dropdb/')
def dropdb():
    db.drop_all()
    return '删除成功'
####################################################
# 数据加密
####################################################
# http://127.0.0.1:5000/addnews/
@blue.route('/addnews/')
def add_news():
    news = News()
    news.title = "王珞丹演新电视剧"
    news.content = "2020年，王珞丹接拍新戏了，震惊！！！"
    news.save()
    return 'Add News Success'

# http://127.0.0.1:5000/getnews/
@blue.route('/getnews/')
def get_news():
    new_list = News.query.all()
    return render_template("news.html", new_list=new_list)