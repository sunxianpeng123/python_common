# encoding: utf-8

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""
from flask import Blueprint

base_blue = Blueprint('base_blue', __name__)

# http://127.0.0.1:5000/
@base_blue.route('/')
def index():
    return 'index'

#######################################################
# 1、单个路由
#######################################################
# 接受输入参数,并指定类型
# int,float 直接写
# string 默认,他是以 / 作为结尾的
# path 会将getpath之后的所有都当做参数
# uuid 只接受uuid格式字符串，唯一码
# any  可以同时指定多种路径，进行限定,类似于穷举,只能写存在于any中的值
# http://127.0.0.1:5000/user/19/
@base_blue.route('/user/<int:id>/')
def users(id):
    print('id = {}, type = {}'.format(id, type(id)))
    return 'user = {}'.format(id)

# http://127.0.0.1:5000/getinfo1/aa/
@base_blue.route('/getinfo1/<string:token>/')
def get_info(token):
    print('token = {}, type = {}'.format(token, type(token)))
    return 'get token = {}'.format(token)

# http://127.0.0.1:5000/getpath/dir1/dir2/
@base_blue.route('/getpath/<path:address>/')
def get_path(address):
    print('path = {}, type = {}'.format(address, type(address)))
    return 'get path = {}'.format(address)

# http://127.0.0.1:5000/getuuid/1e9aec82-34f4-4743-921f-01cd98682077/
@base_blue.route('/getuuid/<uuid:uu>/')
def get_uuid(uu):
    print('uuid = {}, type = {}'.format(uu, type(uu)))
    return 'get uuid = {}'.format(uu)

# http://127.0.0.1:5000/getany/
@base_blue.route('/getany/<any(a,b):an>/')
def get_any(an):
    print('any = {}, type = {}'.format(an, type(an)))
    return 'get any = {}'.format(an)

#######################################################
#2、多个路由
#######################################################
# 可以同时支持 string 和 int 类型
# http://127.0.0.1:5000/getinfo2/aa/
@base_blue.route('/getinfo2/<string:token>/')
@base_blue.route('/gettoken2/<int:token>/')
def get_info2(token):
    print('token = {}, type = {}'.format(token, type(token)))
    return 'get token = {}'.format(token)
