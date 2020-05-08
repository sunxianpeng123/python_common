# -*- coding: utf-8 -*-
# @Time : 2020/5/9 1:15
# @Author : sxp
# @Email : 
# @File : 1_helloworld_1.py
# @Project : python_common

from flask import Flask
from flask_script import Manager
app = Flask(__name__)
manager = Manager(app=app)

@app.route('/')
def helloworld():
    a = 10
    return 'Hello Flask'
    # return '<h1>今天居然下雨了</h1> {}'.format(a)
if __name__ == '__main__':
    manager.run()
    print("running...")
    # 启动方法：usage: 1_helloworld_2.py [-?] {shell,runserver} ...
    # python 1_helloworld_2.py runserver
    # python 1_helloworld_2.py runserver --help 帮助
    # python 1_helloworld_2.py runserver -p 80 -h 0.0.0.0 -d -r --thread