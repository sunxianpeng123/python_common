# encoding: utf-8

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""
import random

from flask import Blueprint, render_template

from App.extentions import db
from App.models.base import Student

ordinary_select_blue = Blueprint('ordinary_select_blue', __name__, template_folder='templates',url_prefix='/select')

###################################################################
# 简单查询
###################################################################
# http://127.0.0.1:5000/select/getstudent/
@ordinary_select_blue.route('/getstudent/')
def get_student():
    # 拿出第一条，没有last函数
    student_1 = Student.query.first()
    # ident 主键
    student_2 = Student.query.get_or_404(ident=2)
    # student_3 = <Student 1>
    # student_3 = <Student 2>
    # get只支持id 不支持其他字段
    student_3 = Student.query.get(2)
    print('student_3 = {}'.format(student_3))
    return 'Get Success'

# http://127.0.0.1:5000/select/getstudents/
@ordinary_select_blue.route('/getstudents/')
def get_students():
    # 取出所有数据
    students = Student.query.all()
    for s in students:
        print(s.name)
    print('student = {}'.format(students))
    return 'Get Success'

###################################################################
# 渲染模板
###################################################################
# http://127.0.0.1:5000/select/render_t/
@ordinary_select_blue.route('/render_t/')
def render_t():
    students = Student.query.all()
    return render_template('students.html', students=students)


