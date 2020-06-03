# -*- coding: utf-8 -*-
# @Time : 2020/5/11 1:26
# @Author : sxp
# @Email : 
# @File : settings.py
# @Project : helloworld
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_uri(dbinfo):
    engine = dbinfo.get('engine') or 'sqlite'
    driver = dbinfo.get('driver') or 'sqlite'
    username = dbinfo.get('username') or ''
    password = dbinfo.get('password') or ''
    host = dbinfo.get('host') or 'localhost'
    port = dbinfo.get('port') or '3306'
    dbname = dbinfo.get('dbname') or ''
    result = '{}+{}://{}:{}@{}:{}/{}'\
        .format(engine, driver, username, password, host, port, dbname)
    return "sqlite:///sqlite.db"

#配置信息
class Config:
    DEBUG = False
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # session使用的secret_key
    SECRET_KEY = 'sxp'
    """ 指定 session 的redis配置，之后存储 session 就在redis中"""
    # SESSION_TYPE = 'redis'
    # SESSION_COOKIE_SECURE = True
    # SESSION_USE_SIGNER = True
    # SESSION_KEY_PREFIX = 'session:'
    # SESSION_REDIS = '127.0.0.1:6379'
    # SESSION_MEMCACHED = '127.0.0.1:11211'

#   开发环境配置
class DevelogConfig(Config):
    DEBUG = True
    dbinfo = {
        'engine':'sqlite',
        # 'driver':'pymysql',
        'driver': '',
        'password':'',
        'host':'',
        'prot':'',
        'dbname':'datatransfer.db'
    }
    SQLALCHEMY_DATABASE_URI = get_uri(dbinfo)

# 测试环境配置
class TestConfig(Config):
    TESTING = True
    dbinfo = {
        'engine':'sqlite',
        # 'driver':'pymysql',
        'driver': '',
        'password':'',
        'host':'',
        'prot':'',
        'dbname':'datatransfer.db'
    }
    SQLALCHEMY_DATABASE_URI = get_uri(dbinfo)

# 演示环境配置
class StagingConfig(Config):
    dbinfo = {
        'engine':'sqlite',
        # 'driver':'pymysql',
        'driver': '',
        'password':'',
        'host':'',
        'prot':'',
        'dbname':'datatransfer.db'
    }
    SQLALCHEMY_DATABASE_URI = get_uri(dbinfo)

# 生产环境配置
class ProductConfig(Config):
    dbinfo = {
        'engine':'sqlite',
        # 'driver':'pymysql',
        'driver': '',
        'password':'',
        'host':'',
        'prot':'',
        'dbname':'datatransfer.db'
    }
    SQLALCHEMY_DATABASE_URI = get_uri(dbinfo)


envs = {
    'develop':DevelogConfig,
    'testing':TestConfig,
    'staging':StagingConfig,
    'product':ProductConfig,
    'default': DevelogConfig
}