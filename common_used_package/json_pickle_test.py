# encoding: utf-8

"""
@author: sunxianpeng
@file: json_pickle_test.py
@time: 2019/12/1 20:04
"""
import json

def dumps():
    # dumps可以格式化所有的基本数据类型为字符串
    list_json = json.dumps([])
    # 数字
    num_json = json.dumps(1)
    # 字符串
    str_json = json.dumps('abc')
    dict = {"name": "Tom", "age": 23}
    dict_json = json.dumps(dict)  # 字典

    print('list json = {}'.format(list_json))
    print('num json = {}'.format(num_json))
    print('str json = {}'.format(str_json))
    print('dict json = {}'.format(dict_json))
    print(type(dict_json))
    with open('test.json','w',encoding='utf-8') as f:
        # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格,美化json格式
        f.write(json.dumps(dict, indent=4))

def dump():
    dict = {"name": "Tom", "age": 24}
    json.dump(dict,open('test.json','w',encoding='utf-8'))

def loads():

    dict_1 = {"name": "Tom", "age": 23}
    dict_json = json.dumps(dict_1)  # 字典
    dict_2 = json.loads(dict_json)
    print('dict_1 = {}'.format(dict_1))
    print('dict_2 = {}'.format(dict_2))

    num_json_1 = json.dumps(1)
    num_json_2 = json.loads(num_json_1)
    print(num_json_2)#1
    print(type(num_json_2))#<class 'int'>

def load():
    dict_json = json.load(open('test.json','r',encoding='utf-8'))
    print(type(dict_json))#<class 'dict'>
    print(dict_json)#{'name': 'Tom', 'age': 24}

if __name__ == '__main__':
    # dumps()
    # dump()
    # loads()
    load()