# encoding: utf-8

"""
@author: sunxianpeng
@file: res.py
@time: 2020/5/16 20:11
"""
from flask import Blueprint, render_template, make_response, Response, abort

response_blue = Blueprint('response_blue', __name__)
#######################################################
# response 相应请求,服务器返回给客户端的数据
#     直接返回Response对象
#     通过make_response（data， code）
#               -data 返回的数据内容
#               -code 状态码
#     返回文本内容
#     返回template模板
#     终止请求
#               abort(404)
#       捕获异常
#               @response_blue.errorhandler(401)
#
#######################################################
# http://127.0.0.1:5000/getresponse1/
@response_blue.route('/getresponse1/')
def get_response1():
    #状态码 修改为400
    return 'response success ',400

# http://127.0.0.1:5000/get_response2/
@response_blue.route('/get_response2/')
def get_response2():
    # 读取模板 相应请求
    result = render_template('Hello.html')
    # result 为string类型的 html内容
    print('result = {}, type = {}'.format(result, type(result)))
    return result

# http://127.0.0.1:5000/get_response3/
@response_blue.route('/get_response3/')
def get_response3():
    # 自己创建请求相应

    # response = <Response 54 bytes [200 OK]>, type = <class 'flask.wrappers.Response'>
    # response = make_response("<h2>班主任说了两个域要去团建都没去</h2>")

    # response = <Response 18 bytes [200 OK]>, type = <class 'flask.wrappers.Response'>
    response = Response('自己造一个DIY')

    print('response = {}, type = {}'.format(response, type(response)))
    return response


# http://127.0.0.1:5000/abort_response/
@response_blue.route('/abort_response/')
def abort_response():
    # 自己创建请求相应

    # response = <Response 54 bytes [200 OK]>, type = <class 'flask.wrappers.Response'>
    # response = make_response("<h2>班主任说了两个域要去团建都没去</h2>")

    # response = <Response 18 bytes [200 OK]>, type = <class 'flask.wrappers.Response'>
    response = Response('自己造一个DIY')

    abort(404)# 终止请求，返回 404 状态码
    print('response = {}, type = {}'.format(response, type(response)))
    return response

# http://127.0.0.1:5000/abort_response/
@response_blue.errorhandler(404)
def handle_error(error):
    # abort(401) 会转向errorhandler，处理401
    print('error  = {}, type = {}'.format(error, type(error)))
    return 'what'

