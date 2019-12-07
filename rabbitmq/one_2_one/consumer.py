# encoding: utf-8

"""
@author: sunxianpeng
@file: consumer.py
@time: 2019/11/25 16:30
"""

import pika
import time
# 四个参数为标准格式
def call_back(ch,method,properties,body):
    print(ch,method,properties)
    # 管道内存对象  内容相关信息  后面讲
    print(" [x] Received %r" % body)
    time.sleep(15)
    # 告诉生成者，消息处理完成
    ch.basic_ack(delivery_tag = method.delivery_tag)





if __name__ == '__main__':
    host = '192.168.199.137'
    port = '5672'
    # 建立实例
    user = 'superrd'
    password = 'superrd'
    ip = '192.168.199.137'
    port = 5672
    credentials = pika.PlainCredentials(user, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        ip, port, '/', credentials))
    # 声明管道
    channel = connection.channel()
    # 为什么又声明了一个‘hello’队列？
    # 如果确定已经声明了，可以不声明。但是你不知道那个机器先运行，所以要声明两次。
    # durable的作用只是把队列持久化。
    channel.queue_declare(queue='hello', exclusive=False, durable=True)
    channel.basic_qos(prefetch_count=1)  # 类似权重，按能力分发，如果有一个消息，就不在给你发
    # 消费消息
    channel.basic_consume(
        # 你要从那个队列里收消息
        queue='hello',
        # 如果收到消息，就调用callback函数来处理消息
        on_message_callback=call_back,
        # 写的话，如果接收消息，机器宕机消息就丢了
        # no_ack=True
        # 一般不写。宕机则生产者检测到发给其他消费者
    )
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()  # 开始消费消息