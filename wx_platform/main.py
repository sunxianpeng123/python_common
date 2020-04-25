# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2020/4/25 17:23
"""
import web
from handle import Handle
urls = ('/shuhong', 'Handle')

class Handle(object):
    def GET(self):
        return "hello, this is handle view"

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()