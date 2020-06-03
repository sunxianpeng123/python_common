# encoding: utf-8

"""
@author: sunxianpeng
@file: __init__.py.py
@time: 2020/5/17 20:37
"""

from App.views.news_view import blue
def init_view(app):
    app.register_blueprint(blue)