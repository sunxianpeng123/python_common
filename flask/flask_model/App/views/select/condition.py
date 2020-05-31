# encoding: utf-8

"""
@author: sunxianpeng
@file: condition.py
@time: 2020/5/23 19:16
"""
from flask import Blueprint, render_template, request
from sqlalchemy import and_, or_, not_

from App.models.base import Student, Cat, Dog, Address, Customer

condition_select_blue = Blueprint('condition_select_blue', __name__, template_folder='templates',url_prefix='/select')

###################################################################
# 条件查询
#           filter：返回BaseQuery对象,__str__属性输出的是这个对象数据的SQL
#               用法
#                       filter(类名.属性.运算符（‘xxx’）)
#                       filter(类名.属性名.数学运算符 值)
#               方法
#                       contains、 startwith、endswith、in_、like、__gt__、__ge__、__lt__、__le__
#                        filter_by:
#                                   根据某个条件筛选
#                                   用在级联数据查询上
#                                   条件语法精准
#                                   条件写为 字段=值
#                        offset：偏移量，跳过多少条
#                        limit: 分页
#                        order_by：根据某个字段排序
#                        get：根据指定的主键获取
#                        first：获取第一条
#                        paginate：
#           级联查询：多表关联查询
#                 获取数据两种方式
#                       手动实现获取
#                       使用关系 relationship 进行级联查询：反向引用
#                 关系
#                       1:1
#                           使用ForeignKey + Unique
#                       1:M
#                           使用ForeignKey
#                       M:N
#                           额外关系表：使用两个ForeignKey
# 逻辑运算
#           注意：筛选功能在flask-sqlalchemy中，如果使用all则只能放在最后


###################################################################
# http://127.0.0.1:5000/select/getcats/
@condition_select_blue.route('/getcats/')
def get_cats():
    #contains、 startwith、endswith、in_、like、__gt__、__ge__、__lt__、__le__
    print("===============一、单个条件==================")
    # 1、获取 id = 2 的猫
    cats = Cat.query.filter(Cat.id.__eq__(2))
    print('cats = {}, type = {}'.format(cats, type(cats)))
    # 2、测试 filter 和 all的顺序功能
    cats2 = Cat.query.filter(Cat.id.__eq__(2)).all()
    print('cats2 = {}, type = {}'.format(cats2, type(cats2)))
    # 3、获取 大于 2  的猫
    cats = Cat.query.filter(Cat.id.__lt__(2))#>
    cats = Cat.query.filter(Cat.id > 2)
    # 4、获取 等于 2  的猫
    cats = Cat.query.filter(Cat.id == 2)
    # 5、获取 name 中包含猫的
    cats = Cat.query.filter(Cat.name.contains('猫'))

    print("===============二、多个条件==================")
    # 6、获取名字包含猫并且 id=2 的猫
    cats = Cat.query.filter(Cat.name.contains('猫')).filter(Cat.id == 2)

    print("===============三、offset 和 limit==================")
    # 注意：offset 和 limit  不区分顺序，都是先执行 offst
    # 7、跳过第一个，取出最多两只猫
    cats = Cat.query.offset(1).limit(2)
    # 8、和 上面的结果相同
    cats = Cat.query.limit(2).offset(1)

    print("===============四、排序==================")
    #注意： order_by 必须在 offset 和 limit 之前
    cats = Cat.query.order_by(Cat.id).limit(2)
    # return 'Get Cats Success'
    return render_template('cats.html', cats=cats)



# http://127.0.0.1:5000/select/getdogs/?page=2&per_page=5
@condition_select_blue.route('/getdogs/')
def get_dogs():
    print("===============六、使用 offset 和 limit 实现数据查询分页==================")
    # 1、 使用 offset 和 limit 实现数据查询分页
    #page 表示参数的key, 1 表示默认第一页, type 表示接收到的参数类型
    page = request.args.get('page',1, type=int)
    per_page = request.args.get('per_page', 4, type=int)
    #数据总共分成 per_page*(page - 1) 页，获取第 page 页，每页条数为 per_page,
    dogs = Dog.query.offset(per_page*(page - 1)).limit(per_page)
    return render_template('dogs.html',dogs=dogs)

# 不写分页的参数 ：默认每页 20 条
# http://127.0.0.1:5000/select/getdogswithpaginate
# 写分页的参数
# http://127.0.0.1:5000/select/getdogswithpaginate/?per_page=5
# http://127.0.0.1:5000/select/getdogswithpaginate/?page=2&per_page=5
@condition_select_blue.route('/getdogswithpaginate/')
def get_dogs_with_paginate():
    print("===============七、使用paginate实现数据查询分页================")
    per_page = request.args.get('per_page', 4, type=int)
    pagination = Dog.query.paginate()
    dogsItem = pagination.items
    return render_template('dogs.html',dogsItem=dogsItem, pagination=pagination, per_page=per_page)

# http://127.0.0.1:5000/select/getcatsfilterby
@condition_select_blue.route('/getcatsfilterby/')
def get_cats_filter_by():
    #直接写属性名字的条件
    cats = Cat.query.filter_by(id=2)
    return render_template('cats.html', cats=cats)

##########################################################
# 级联查询
##########################################################

# http://127.0.0.1:5000/select/getcustomer/?a_id=7
@condition_select_blue.route('/getcustomer/')
def get_customer():
    """获取地址对应的消费者"""
    # 指定 address 的id， 然后获取指定 address.id对应的customer
    a_id = request.args.get('a_id', type=int)
    address = Address.query.get_or_404(a_id)
    customer = Customer.query.get(address.customer_id)
    return customer.name

# http://127.0.0.1:5000/select/getaddress/?c_id=7
@condition_select_blue.route('/getaddress/')
def get_address():
    """获取"""
    # 指定 address 的id， 然后获取指定 address.id对应的customer
    c_id = request.args.get('c_id', type=int)
    customer = Customer.query.get(c_id)
    # model中不加addresses = db.relationship('Address', backref='customer', lazy=True)可以这么写
    # addresses = Address.query.filter_by(customer_id = customer.id)
    # model中添加    addresses = db.relationship('Address', backref='customer', lazy=True)，可以这么写
    # backref反向引用，只有当两张表通过外键关联时才可以这样使用，否则不可以
    addresses = customer.addresses
    print('addresses = {},type = {}'.format(addresses, type(addresses)))
    return render_template('address.html', addresses=addresses)

##########################################################
# 逻辑运算
##########################################################
# http://127.0.0.1:5000/select/getaddresswithlogical
@condition_select_blue.route('/getaddresswithlogical/')
def get_addresses_with_logical():
    """获取地址对应的消费者"""
    # and_ 与
    # addresses = Address.query.filter(Address.customer_id.__eq__(1)).filter(Address.position.endswith('4'))
    addresses = Address.query.filter(and_(Address.customer_id.__eq__(1),Address.position.endswith('4')))
    # or_ 或
    addresses = Address.query.filter(or_(Address.customer_id.__eq__(1),Address.position.endswith('4')))
    # not_ 非
    addresses = Address.query.filter(and_(Address.customer_id.__eq__(1)))
    addresses = Address.query.filter(not_(or_(Address.customer_id.__eq__(1),Address.position.endswith('4'))))
    return render_template('address.html', addresses=addresses)





