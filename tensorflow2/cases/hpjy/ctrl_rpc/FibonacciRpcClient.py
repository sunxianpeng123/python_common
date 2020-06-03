# encoding: utf-8

"""
@author: sunxianpeng
@file: FibonacciRpcClient.py
@time: 2020/6/3 11:05
"""
import pika
import logging
import json
import uuid
import time

from ctrl_rpc import get_this_file_name

this_file_name = get_this_file_name()
logger = logging.getLogger(this_file_name)


class FibonacciRpcClient(object):
    def __init__(self, mq_info):
        logger.info("client to server")
        self.credentials = pika.PlainCredentials(mq_info["user"], mq_info['password'])
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(mq_info['ip'], mq_info['port'], '/', self.credentials))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def getSendJson(self, images, method, uid, id):
        sendJson = {};
        sendJson['uid'] = uid
        sendJson['id'] = id
        sendJson['method'] = method
        sendJson['params'] = images
        strjson = json.dumps(sendJson)
        bjson = strjson.encode(encoding='utf-8')
        return bjson

    def sendToServer(self, image, method, queue):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.sendJson = self.getSendJson(image, method, self.corr_id, self.corr_id)

        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=self.sendJson)
        i = 0
        while self.response is None:
            i = i + 1
            self.connection.process_data_events()
            if i > 3000:
                logging.info("time out")
                break
            time.sleep(0.01)
        return self.response

    def sendImage(self, image, method, queue, mq_info):
        try:
            result = self.sendToServer(image, method, queue)
        except:
            self.__init__(mq_info)
            result = self.sendToServer(image, method, queue)
        return result
