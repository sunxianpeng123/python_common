# -*- coding: utf-8 -*-
# @Time : 2020/5/10 2:25
# @Author : sxp
# @Email : 
# @File : helloworld.py
# @Project : python_common

from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    return 'hello world'


if __name__ == '__main__':
    # 启动nginx
    # /usr/local/nginx/sbin/nginx
    # 启动flask项目
    # python helloworld.py
    # 访问页面
    # http://101.132.128.193/
    app.run(port=5000)