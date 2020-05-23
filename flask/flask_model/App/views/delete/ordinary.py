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

ordinary_delete_blue = Blueprint('ordinary_delete_blue', __name__,template_folder='templates', url_prefix='/delete')

###################################################################
#
###################################################################
# http://127.0.0.1:5000/delete/deletestudent/
@ordinary_delete_blue.route('/deletestudent/')
def delete_student():
    student = Student.query.first()
    db.session.delete(student)
    db.session.commit()

    return 'Delete Success'




