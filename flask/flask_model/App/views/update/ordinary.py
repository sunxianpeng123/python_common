# encoding: utf-8

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""
import random

from flask import Blueprint

from App.extentions import db
from App.models.base import Student

ordinary_update_blue = Blueprint('ordinary_update_blue', __name__, template_folder='templates',url_prefix='/update')

###################################################################
#
###################################################################
# http://127.0.0.1:5000/update/updatestudent/
@ordinary_update_blue.route('/updatestudent/')
def update_student():
    student = Student.query.first()
    print("student name = {}".format(student.name))
    student.name = 'Tom'
    student.save()
    return 'Get Success'



