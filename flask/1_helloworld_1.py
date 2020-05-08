# -*- coding: utf-8 -*-
# @Time : 2020/5/9 1:15
# @Author : sxp
# @Email : 
# @File : 1_helloworld_1.py
# @Project : python_common

from flask import Flask
app = Flask(__name__)

@app.route('/')
def helloworld():
    return 'Hello Flask'

@app.route('/html')
def hellohtml():
    a = 10
    return '<h1>今天居然下雨了</h1> {}'.format(a)

#
# jinjia2 模板引擎
# werkzeug wsgi 工具集
# itsdangerous 给予Django的签名模板
#
if __name__ == '__main__':
    #
    import os
    os.environ.setdefault("Flask_ENV", 'develop')
    app.run(debug=True, port=80, )
    print("running...")