# encoding: utf-8

"""
@author: sunxianpeng
@file: model.py
@time: 2020/5/23 14:42
"""
from App.extentions import db

##########################################################
#    __tablename__ = "Student"  指定表名
##########################################################
class Student(db.Model):
    # 默认为 student，可以修改表名
    __tablename__ = "Student"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(16), primary_key=False)

    def save(self):
        db.session.add(self)
        db.session.commit()

class User(db.Model):
    # 默认为user，可以修改表名
    __tablename__ = "UserModel"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)#自增
    name = db.Column(db.String(16),unique=True)
    description = db.Column(db.String(128),nullable=True)

    def save(self):
        db.session.add(self)
        db.session.commit()
##########################################################
# 模型继承
#       默认继承并不会报错，他会将多个模型的数据映射到一张表中，导致数据混乱，不能满足基本使用
#       抽象的模型是不会再数据库中产生映射的  ，__abstract__ = True 可以指定模型为抽象模型
# 模型文档
#       可以学习 flask-sqlalchemy、 sqlalchemy 的模型文档
# 模型之间的关系
#       可以用继承来体现
# 项目中的数据库优化
#       怎么连接
#       连接多少个
#       Django 和 Flask 默认都是有数据库连接池的


##########################################################
class Animal(db.Model):
    __abstract__ = True
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(16))
    def save(self):
        db.session.add(self)
        db.session.commit()

class Dog(Animal):
    legs = db.Column(db.Integer, default=4)

class Cat(Animal):
    eat = db.Column(db.String(32), default="小鱼干")
##########################################################
# 级联数据， 一对多的关系模型的构建
##########################################################
class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(16))

    addresses = db.relationship('Address', backref='customer', lazy=True)

    def save(self):
        db.session.add(self)
        db.session.commit()

class Address(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    position = db.Column(db.String(128))
    # 外键
    customer_id = db.Column(db.Integer, db.ForeignKey(Customer.id),nullable=False)
    def save(self):
        db.session.add(self)
        db.session.commit()