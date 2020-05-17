# encoding: utf-8

"""
@author: sunxianpeng
@file: requestion.py
@time: 2020/5/16 18:55
"""
from flask import Blueprint, request

request_blue = Blueprint('request_blue', __name__)

#######################################################
#3、flask request
    # flask四大内置对象：
#       1、Request：request ,若要使用直接导入即可
#       2、Session：session
#       3、G: g
#       4、Config
#             在模板中 ：config
#             在python 代码中 app.config

# """ 需要搞清楚 request、session的原理"""
#######################################################
# http://127.0.0.1:5000/getrequest/
# python manager.py runserver -r -d -h 0.0.0.0
@request_blue.route('/getrequest/', methods=['get', 'post', 'delete'])
def get_request():
    """"""
    """1、接收url中的参数"""
    print('request.method = {}'.format(request.method))  # 获取访问方式 GET
    print('request.url = {}'.format(request.url))  # 获取url http://127.0.0.1:5000/req?id=1&name=wl
    print('request.cookies = {}'.format(request.cookies))  # 获取cookies {}
    print('request.path = {}'.format(request.path))  # 获取访问路径 /req
    print('request.args = {}'.format(request.args))  # 获取url传过来的值  ImmutableMultiDict([('id', '1'), ('name', 'wl')])
    # print('request.args.get("id") = {}'.format(request.args.get("id")))  # get获取id  1
    # print('request.args["name"] = {}'.format(request.args["name"]))  # 索引获取name wl
    # print('request.args.to_dict() = {}'.format(request.args.to_dict()))  # #获取到一个字典 {'id': '1', 'name': 'wl'}
    """2、接收form中的参数"""
    # <!DOCTYPE html>
    # <html lang="en">
    # <head>
    #     <meta charset="UTF-8">
    #     <title>Title</title>
    # </head>
    # <body>
    # <form action="" method="post">
    #     <input type="text" name="username">
    #     <input type="password" name="pwd">
    #     <input type="submit" value="登陆">
    # </form>
    # </body>
    # </html>
    # 后端接收到的参数
    #
    # @app.route("/login",methods=["POST","GET"])
    # def login():
    #     print(request.form) # 格式 ImmutableMultiDict([('username', '123'), ('pwd', '123')])
    #     print(request.form.to_dict()) # 格式 {'username': '123', 'pwd': '123'}
    #     return render_template("login.html")
    """3、接收文件"""
    # request.files: 接收文件
    """4、json data values"""

    # 判断请求方法
    if request.method == 'get':
        return 'get success'
    elif request.method.lower() == 'post':
        return 'post success'
    else:
        return '%s success' % request.method

#######################################################
# 请求获取url中的参数
# url中的请求参数 ：
#       query_string,
#       query_params，
#       get 请求参数，
# form:
#       表单数据，
#       post请求参数（直接支持put， patch），
#######################################################
@request_blue.route('/args/', methods = ['get', 'post', 'delete', 'put', 'patch'])
def get_args():
    print(request.method)
    if request.method.lower() == 'get':
        # http://127.0.0.1:5000/args/
        # http://127.0.0.1:5000/args/?user=rock&password=rock1204
        # http://127.0.0.1:5000/args/?user=rock&password=rock1204&password=rock1205
        # 多个密码值
        print("我是 Get 请求================")
        print('request.args = {}, type = {}'.format(request.args, type(request.args)))
        print('request.args.get("user") = {}'.format(request.args.get("user")))  # get获取id  1
        print('request.args["password"] = {}'.format(request.args["password"]))  # 索引获取name wl
        print('request.args.get("password") = {}'.format(request.args.get("password")))  # get获取id  1

    if request.method.lower() == 'post':
        # 同样可以获取 到 url中的数据
        # postman 下，bady 指定 msg：sleeping， name：tom
        # http://127.0.0.1:5000/args/?user=rock&password=rock1204&password=rock1205
        # 多个密码值
        print("我是 Post 请求================")
        print('request.args = {}, type = {}'.format(request.args, type(request.args)))
        print('request.args.get("user") = {}'.format(request.args.get("user")))  # get获取id  1
        print('request.args["password"] = {}'.format(request.args["password"]))  # 索引获取name wl
        print('request.args.get("password") = {}'.format(request.args.get("password")))  # get获取id  1
        print('request.form = {}, type = {}'.format(request.form, type(request.form)))# 2、接收form中的参数

    return 'args success'

