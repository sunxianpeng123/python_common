# encoding: utf-8

"""
@author: sunxianpeng
@file: cookies.py
@time: 2020/5/16 20:56
"""

#       由于HTTP是一种无状态的协议，服务器单从网络连接上无从知道客户身份。怎么办呢？就给客户端们颁发一个通行证吧，每人一个，
# 无论谁访问都必须携带自己通行证。这样服务器就能从通行证上确认客户身份了。这就是Cookie的工作原理。
#       Cookie实际上是一小段的文本信息。客户端请求服务器，如果服务器需要记录该用户状态，就使用response向客户端浏览器颁发一个Cookie。
# 客户端浏览器会把Cookie保存起来。当浏览器再请求该网站时，浏览器把请求的网址连同该Cookie一同提交给服务器。服务器检查该Cookie，以此来辨认用户状态。服务器还可以根据需要修改Cookie的内容。


from flask import Blueprint, request, render_template, Response, session

cookies_blue = Blueprint('cookies_blue', __name__)


#######################################################
# 看完这篇 Session、Cookie、Token，和面试官扯皮就没问题了
# https://blog.csdn.net/qq_36894974/article/details/105322171?utm_medium=distribute.pc_feed.none-task-blog-alirecmd-19.nonecase&depth_1-utm_source=distribute.pc_feed.none-task-blog-alirecmd-19.nonecase&request_id=
# 会话技术
#     跨请求共享数据
#     出现原因
#               web开发中http都是短连接（）
#               http请求是无状态的
#               请求从request开始到response结束
# 两种会话技术
#         cookies：
#               客户端会话技术
#               数据存储在客户端
#               数据格式： key-value
#               flask中cookies默认支持中文
#         Session：
#               服务端会话技术
#               数据存储在服务器
#               key-value格式
#               flask中
#                   将session存储在cookie中!!
#                   对数据进行序列化
#                   进行了base64
#                   还进行了zlib压缩
#                   还传递了hash
#
#         Token
# Flask-Session
#         实现了服务端session
#         将数据存储在服务店，将数据对应的key存储在cookie中
#         RedisSessionInterface
#                   save_session
#                           将数据进行了pickle序列化
#######################################################

#######################################################
# 1、使用cookie
#######################################################

# http://127.0.0.1:5000/login1/
@cookies_blue.route('/login1/', methods=['get', 'post'])
def login1():
    if request.method.lower() == 'get':

        return render_template('login_cookie.html')
    elif request.method.lower() == 'post':
        username = request.form.get('username')
        # username存储在cookies中
        response = Response("登录成功 {}".format(username))
        response.set_cookie('username', username)
        return response


# http://127.0.0.1:5000/getcookies1/
@cookies_blue.route('/getcookies1/')
def get_cookies1():
    # 获取login页面 cookies 里面的 username
    username = request.cookies.get('username')
    print("username = {}".format(username))
    return "欢迎回来 {}".format(username)


#######################################################
# 2、使用 session
# 将session存储在cookie中!!
#######################################################

# http://127.0.0.1:5000/login2/
@cookies_blue.route('/login2/', methods=['get', 'post'])
def login2():
    if request.method.lower() == 'get':
        return render_template('login_session.html')
    elif request.method.lower() == 'post':
        username = request.form.get('username')
        # username存储在cookies中
        response = Response("登录成功 {}".format(username))
        session['username'] = username
        session['password'] = '110'
        # session = <SecureCookieSession {'username': 'sxs', 'password': '110'}>, type = <class 'werkzeug.local.LocalProxy'>
        print('session = {}, type = {}'.format(session, type(session)))
        return response


# http://127.0.0.1:5000/getcookies2/
@cookies_blue.route('/getcookies2/')
def get_cookies2():
    # 获取login页面 cookies 里面的 username
    username = session.get('username')
    print("username = {}".format(username))
    return "欢迎回来 {}".format(username)

#######################################################
# 3、flask-session
#######################################################

# http://127.0.0.1:5000/login2/
@cookies_blue.route('/login3/', methods=['get', 'post'])
def login3():
    if request.method.lower() == 'get':
        return render_template('login_session.html')
    elif request.method.lower() == 'post':
        username = request.form.get('username')
        # username存储在cookies中
        response = Response("登录成功 {}".format(username))
        session['username'] = username
        session['password'] = '110'
        # session = <SecureCookieSession {'username': 'sxs', 'password': '110'}>, type = <class 'werkzeug.local.LocalProxy'>
        print('session = {}, type = {}'.format(session, type(session)))
        return response

# http://127.0.0.1:5000/getcookies3/
@cookies_blue.route('/getcookies3/')
def get_cookies3():
    # 获取login页面 cookies 里面的 username
    username = session.get('username')
    print("username = {}".format(username))
    return "欢迎回来 {}".format(username)