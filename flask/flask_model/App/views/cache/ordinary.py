# encoding: utf-8

"""
@author: sunxianpeng
@file: base.py
@time: 2020/5/16 18:44
"""
import random

from flask import Blueprint, render_template
from sqlalchemy import *

from App.extentions import db, cache
from App.models.base import Student, Address

cache_select_blue = Blueprint('cache_select_blue', __name__, template_folder='templates',url_prefix='/cache')

###################################################################

###################################################################
# http://127.0.0.1:5000/cache/getaddresswithlogical
@cache_select_blue.route('/getaddresswithlogical/')
@cache.cached(timeout=50)
def get_addresses_with_logical():
    """获取地址对应的消费者"""
    # and_ 与
    # addresses = Address.query.filter(Address.customer_id.__eq__(1)).filter(Address.position.endswith('4'))
    # addresses = Address.query.filter(and_(Address.customer_id.__eq__(1),Address.position.endswith('4')))
    # or_ 或
    addresses = Address.query.filter(or_(Address.customer_id.__eq__(1),Address.position.endswith('4')))
    # not_ 非
    # addresses = Address.query.filter(and_(Address.customer_id.__eq__(1)))
    addresses = Address.query.filter(not_(or_(Address.customer_id.__eq__(1),Address.position.endswith('4'))))

    print("从数据库中查询数据")
    return render_template('address.html', addresses=addresses)


