# -*- coding: utf-8 -*-
# @Time : 2020/6/4 1:56
# @Author : sxp
# @Email : 
# @File : SpiderLearn.py
# @Project : flask_datatransfer
import requests


def get_data(url):
    response = requests.get(url)
    print(response.content.decode('utf-8'))

if __name__ == '__main__':
    url = "http://127.0.0.1:5000/getnews/"
    get_data(url)