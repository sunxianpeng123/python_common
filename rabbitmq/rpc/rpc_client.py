# encoding: utf-8

"""
@author: sunxianpeng
@file: rpc_client.py
@time: 2019/11/27 11:32
"""

import pika
import uuid
import time
class FibonacciRpcClient(object):
    def __init__(self,user,password,ip,port):
        self.credentials = pika.PlainCredentials(user, password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            ip, port, '/', self.credentials))
        self.channel = self.connection.channel()
        # 声明一个队列,durable参数声明队列持久化,把消费者和queue绑定起来
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,# 只要一收到消息就调用on_response
            auto_ack=True
        )  # 收这个queue的消息

    def on_response(self,ch,method,props,body):
        # 如果收到的ID和本机生成的相同，则返回的结果就是我想要的指令返回的结果
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, n):
        self.response = None  # 初始self.response为None
        self.corr_id = str(uuid.uuid4())  # 随机唯一字符串
        self.channel.basic_publish(
                exchange='',
                routing_key='rpc_queue',  # 发消息到rpc_queue
                properties=pika.BasicProperties(  # 消息持久化
                    reply_to = self.callback_queue,  # 让服务端命令结果返回到callback_queue
                    correlation_id = self.corr_id,  # 把随机uuid同时发给服务器
                ),
                body=str(n)
        )
        while self.response is None:  # 当没有数据，就一直循环
            # 启动后，on_response函数接到消息，self.response 值就不为空了
            self.connection.process_data_events()  # 非阻塞版的start_consuming()
            # print("no msg……")
            # time.sleep(0.5)
        # 收到消息就调用on_response
        return int(self.response)


if __name__ == '__main__':
    user = 'superrd'
    password = 'superrd'
    ip = '192.168.199.137'
    port = 5672
    fibonacci_rpc = FibonacciRpcClient(user,password,ip,port)
    print(" [x] Requesting fib(7)")
    response = fibonacci_rpc.call(7)
    print(" [.] Got %r" % response)
