# encoding: utf-8

"""
@author: sunxianpeng
@file: server.py
@time: 2020/6/4 17:29
"""

##############################################################
# Log
##############################################################
import json
import time

import pika

from blood_recognition import check_device
from ctrl_rpc import get_this_file_name, get_logger
from ctrl_rpc.FibonacciRpcClient import FibonacciRpcClient
from ctrl_rpc.settings import get_mq_info
from ctrl_rpc.utils import imageDecode

log_path = 'logs/'
today_datetime = time.strftime("%Y%m%d%H%M%S", time.localtime())
log_name = log_path + 'logs_' + str(today_datetime) + '.log'
log_file = log_name
this_file_name = get_this_file_name()
logger = get_logger(log_file, this_file_name)
logger.info('start')

check_device()
##############################################################
# rabbit-mq info
##############################################################
conf_path = "conf/image_calssify_client_server_rpc_lol.conf"
mq_info = get_mq_info(conf_path)
print(mq_info)

##############################################################
# pika
##############################################################
credentials = pika.PlainCredentials(mq_info['user'], mq_info['password'])
connection = pika.BlockingConnection(pika.ConnectionParameters(mq_info['ip'], mq_info['port'], '/', credentials))
channel = connection.channel()
channel.queue_declare(queue=mq_info['mq_queue_name'], auto_delete=True, exclusive=False, durable=False)
fibonacci_rpc = FibonacciRpcClient(mq_info)


def on_request(ch, method, props, body):
    try:
        logger.info("-----------------get message----------")
        strJson = body.decode("utf-8")
        jsonResult = json.loads(strJson)
        strMethod = jsonResult['method']
        image = jsonResult["params"][0]

        result = imageDecode(image, mq_info, image)
        result = str(result)
        if len(result) > 6:
            result = result.replace("'", "\"")
        strResult = json.dumps(result)
        logger.info(strResult)
        b = strResult.encode(encoding='utf-8')
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body=b)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print("error", e)
        logger.error("error")
        quit()


channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=mq_info['mq_queue_name'], on_message_callback=on_request)

logger.info(" [x] Awaiting RPC requests")
channel.start_consuming()
