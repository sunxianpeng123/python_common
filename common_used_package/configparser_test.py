# encoding: utf-8

"""
@author: sunxianpeng
@file: configparser_test.py
@time: 2019/12/1 20:34
"""
import configparser

def get_config(type='ini'):
    """读取文件形式或者字典形式获取configparser的实例类config"""
    config = configparser.ConfigParser()
    if type == 'ini':
        config.read(config_path)
    if type == 'dict':
        config.read_dict({'section1': {'key1': 'value1','key2': 'value2','key3': 'value3'},
                          'section2': {'keyA': 'valueA','keyB': 'valueB',  'keyC': 'valueC'},
                          'section3': {'foo': 'x', 'bar': 'y','baz': 'z'}})
    return config

def foreach_config(config):
    print("######################查找配置文件中的信息###########################")
    # 1、获取所有 sections
    sections = config.sections()# # 注意会过滤掉[DEFAULT]
    i = 0
    for section in sections:
        print('第 {} 个主题= {}下的配置'.format(i,section))

        print("**********1、获取指定 section 的 keys & values*******")
        items = config.items(section)
        for tup_kv in items:
            print('主题 = {} 下的 第一个键值对为 = {}'.format(section,tup_kv))#('serveraliveinterval', '45')
            break

        print("**********2、获取指定 section 的 keys*******")
        keys = config.options(section)
        for key in keys:
            print('主题 = {} 下的 第一个键为 = {}'.format(section,key))
            break

        print("**********3、获取指定 key 的 value*******")
        value_1 = config[section]['user']
        value_2 = config.get(section,'user')#Tom
        print('主题 = {} 下，key 为 user 的 值 = {}'.format(section,value_1))

        # 获取指定 key 的 value
        i += 1
        break

def check_add_delete_config_info(config):
    """检查、添加、删除配置文件中的信息"""
    print("#####################检查、添加、删除配置文件中的信息#########################")
    print("**********1、检查是否在配置文件中*********")
    section_check_1 = 'DEFAULT' in config
    # has_section()对于检查 DEFAULT 无效，对其他有效，可以和 in 区别使用
    section_check_2 = config.has_section('DEFAULT')
    key_check_1 = 'test' in config['DEFAULT']
    key_check_2 = config.has_option('section_test', 'test')
    value_check = 'Tom' in config['bitbucket.org']['User']
    print("check section default 1 = {}".format(section_check_1))
    print("check section default 2 = {}".format(section_check_2))
    print("check key test of DEFAULT = {}".format(key_check_2))
    print("check value Tom of bitbucket.org = {}".format(value_check))

    print("**********2、向配置文件中添加配置*********")
    config.add_section('Section_1')
    config.set('Section_1', 'key_1', 'value_1')  # 注意键值是用set()方法
    config.write(open('configparser.ini', 'w'))  # 一定要写入才生效

    print("**********3、删除配置文件中的内容********")
    config.remove_option('Section_1', 'key_1')
    config.remove_section('Section_1')
    # config.clear()  # 清空除[DEFAULT]之外所有内容
    config.write(open('configparser.ini', 'w'))



if __name__ == '__main__':
    config_path = 'configparser.ini'# 配置文件的路径
    config = get_config(type='ini')
    foreach_config(config)
    check_add_delete_config_info(config)
