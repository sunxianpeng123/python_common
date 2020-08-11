# encoding: utf-8

"""
@author: sunxianpeng
@file: student.py
@time: 2020/5/17 20:41
"""
from flask import Blueprint, render_template

template_view_blue = Blueprint('template_view_blue', __name__)

#######################################################
# 模板中的控制语句
#  for循环语句
#       可以获取循环信息
#       loop.first
#       loop.last
#       loop.index loop.index0
#       loop.revindex loop.revindex
#  if语句
# 过滤器
#       语法 {{ 变量|过滤器|过滤器|...}}
#       capitalize 驼峰命名法
#       lower
#       upper
#       title
#       trim
#       reverse
#       format
#       striptags 渲染之前，将值中的标签去掉
#       。。。。
#######################################################
# http://127.0.0.1:5000/students/
@template_view_blue.route('/students/', methods=['get', 'post', 'delete'])
def get_student():
    students = ['我是学生: {}'.format(i) for i in range(10)]
    msg = 'helloWorld'
    a = 5
    b = 5
    return render_template('control_statement.html', students=students, msg=msg, a=a, b=b)

#######################################################
# html 模板继承
#       block快操作
#           在父模板中预留位置，在字幕版中填充
#       extends
#            {% extends 'user/user_register.html' %}
#             继承模板，可以实现 追加和覆盖的效果
#            加上 {{ super() }} 表示在父 html上追加，不加{{ super() }} 表示覆盖父html的内容
#       include
#           将其他html包含进来
#       宏定义marco:使前端拥有了编程能力
#            可以在模板中定义函数
#            宏定义可导入:{% from '' import xxx %}
#######################################################
# block快操作, include, 宏定义marco
# http://127.0.0.1:5000/userregister/
@template_view_blue.route('/userregister/', methods=['get', 'post', 'delete'])
def user_register():
    return render_template('user/user_register.html', title="用户注册")

# extends
# http://127.0.0.1:5000/userregister2/
@template_view_blue.route('/userregister2/', methods=['get', 'post', 'delete'])
def user_register2():
    return render_template('user/user_register2.html', title="用户注册2")

