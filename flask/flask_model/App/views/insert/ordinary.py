# encoding: utf-8

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""
import random

from flask import Blueprint
from sqlalchemy import text

from App.extentions import db
from App.models.base import Student, Cat, Dog, Customer, Address

ordinary_insert_blue = Blueprint('ordinary_insert_blue', __name__,template_folder='templates', url_prefix='/insert')

# http://127.0.0.1:5000/insert/
@ordinary_insert_blue.route('/')
def index():
    return 'index'

##########################################################
#
##########################################################

# http://127.0.0.1:5000/insert/createdb/
@ordinary_insert_blue.route('/createdb/')
def createdb():
    db.create_all()
    return '创建db成功'

# http://127.0.0.1:5000/insert/dropdb/
@ordinary_insert_blue.route('/dropdb/')
def dropdb():
    db.drop_all()
    return '删除成功'

##########################################################
# 添加学生 数据
##########################################################

# http://127.0.0.1:5000/insert/addstudent
@ordinary_insert_blue.route("/addstudent/")
def add_student():
    student = Student()
    student.name = "小花%d" % random.randrange(1000)
    student.save()
    print("db.session = {} ， type = {}".format(db.session, type(db.session)))
    return "Add Student Seccess"

# http://127.0.0.1:5000/insert/addstudents
@ordinary_insert_blue.route("/addstudents/")
def add_students():
    students = []
    for i in range(5):
        student = Student()
        student.name = "小明%d" % i
        students.append(student)
    db.session.add_all(students)
    db.session.commit()
    print("db.session = {} ， type = {}".format(db.session, type(db.session)))
    return "Add Students Seccess"

##########################################################
#
##########################################################
# http://127.0.0.1:5000/insert/addcat
@ordinary_insert_blue.route("/addcat/")
def add_cat():
    cat = Cat()
    cat.name = "加菲猫"
    cat.eat = "猫粮"
    cat.save()
    return "Add Cat Seccess"

# http://127.0.0.1:5000/insert/adddog
@ordinary_insert_blue.route("/adddog/")
def add_dog():
    dog = Dog()
    dog.name = "傻狗"
    dog.legs = 4
    dog.save()
    return "Add Dog Seccess"


# http://127.0.0.1:5000/insert/addcustomer
@ordinary_insert_blue.route("/addcustomer/")
def add_customer():
    customer = Customer()
    customer.name = "剁手党%d" % random.randrange(1000)
    customer.save()
    return "Add Customer Seccess %s" % customer.name

# http://127.0.0.1:5000/insert/addaddress
@ordinary_insert_blue.route("/addaddress/")
def add_address():
    address = Address()
    address.position = "上海市徐汇区%d" % random.randrange(10000)
    # 根据id排序，取出最后一个，即拿出最新的一个来
    address.customer_id = Customer.query.order_by(text("-id")).first().id
    address.save()
    return "Add Address Seccess %s" % address.position