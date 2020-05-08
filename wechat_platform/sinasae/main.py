# encoding: utf-8

"""
@author: sunxianpeng
@file: autoResponse.py
@time: 2020/5/2 20:46
"""


# 最简单的Hello World， 会给收到的每一条信息回复 Hello World

import werobot
import os

with open("test.txt","w") as f:
    f.write("test")
f.close()


robot = werobot.WeRoBot(token='aliyun')
@robot.handler
def hello(message):
    return 'Hello World!'

# 让服务器监听在 0.0.0.0:80
robot.config['HOST'] = '0.0.0.0'
robot.config['PORT'] = 80
robot.run()

# import os
# import tornado.httpserver
# import tornado.ioloop
# import tornado.options
# import tornado.web
# from tornado import gen
# from tornado.httpclient import AsyncHTTPClient
#
# class MainHandler(tornado.web.RequestHandler):
#
#     @gen.coroutine
#     def get(self):
#         http_client = AsyncHTTPClient()
#         response = yield http_client.fetch("http://www.sinacloud.com")
#         self.set_header('content-type', 'text/plain')
#         self.write('Hello, World! ' + str(response.body[:100]))
#
# application = tornado.web.Application([
#     (r"/", MainHandler),
# ])
#
# if __name__ == "__main__":
#     tornado.options.parse_command_line()
#     http_server = tornado.httpserver.HTTPServer(application)
#     http_server.listen(5050 or os.environ['PORT'])
#     tornado.ioloop.IOLoop.current().start()