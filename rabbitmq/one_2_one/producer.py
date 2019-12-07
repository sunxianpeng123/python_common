# encoding: utf-8

"""
@author: sunxianpeng
@file: sender.py
@time: 2019/11/25 16:27
"""


import pika

if __name__ == '__main__':
    user = 'superrd'
    password = 'superrd'
    ip = '192.168.199.137'
    port = 5672
    credentials = pika.PlainCredentials(user, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        ip, port, '/', credentials))
    # 声明一个管道，在管道里发消息
    channel = connection.channel()
    # 在管道里声明queue,durable的作用只是把队列持久化。
    channel.queue_declare(queue='hello',durable=True)
    # RabbitMQ a message can never be sent directly to the queue, it always needs to go through an exchange.
    channel.basic_publish(exchange='',
                          routing_key='hello',  # queue名字
                          body='Hello World!',
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          )
                          )  # 消息内容
    print(" [x] Sent 'Hello World!'")
    # connection.close()  # 队列关闭