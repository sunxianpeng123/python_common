# encoding: utf-8

"""
@author: sunxianpeng
@file: rpc_server.py
@time: 2019/11/27 11:48
"""
import pika
import time

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def on_request(ch, method, props, body):
    n = int(body)
    print(" [.] fib(%s)" % n)
    response = fib(n)
    ch.basic_publish(
            exchange='',  # 把执行结果发回给客户端
            routing_key=props.reply_to,  # 客户端要求返回想用的queue
            # 返回客户端发过来的correction_id 为了让客户端验证消息一致性
            properties=pika.BasicProperties(correlation_id = props.correlation_id),
            body=str(response)
    )
    ch.basic_ack(delivery_tag = method.delivery_tag)  # 任务完成，告诉客户端


if __name__ == '__main__':
    user = 'superrd'
    password = 'superrd'
    ip = '192.168.199.137'
    port = 5672
    credentials = pika.PlainCredentials(user, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        ip, port, '/', credentials))

    channel = connection.channel()
    channel.queue_declare(queue='rpc_queue')  # 声明一个rpc_queue ,

    channel.basic_qos(prefetch_count=1)
    # 在rpc_queue里收消息,收到消息就调用on_request
    # 消费消息
    channel.basic_consume(
        # 你要从那个队列里收消息
        queue='rpc_queue',
        # 如果收到消息，就调用callback函数来处理消息
        on_message_callback=on_request,
        # 写的话，如果接收消息，机器宕机消息就丢了
        # auto_ack=True
        # 一般不写。宕机则生产者检测到发给其他消费者
    )
    print(" [x] Awaiting RPC requests")
    channel.start_consuming()