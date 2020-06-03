# encoding: utf-8

"""
@author: sunxianpeng
@file: server.py
@time: 2020/6/3 17:30
"""
import json
import time

import cv2
import pika
import keras
import numpy as np
import tensorflow as tf
from kill_ation_detection import get_this_file_name, get_logger
from kill_ation_detection.settings import get_mq_info

##############################################################
# Log
##############################################################
from kill_ation_detection.utils import imageDecode

log_path = 'logs/'
# today_datetime = time.strftime("%Y%m%d", time.localtime())
today_datetime = time.strftime("%Y%m%d%H%M%S", time.localtime())
log_name = log_path + 'logs_' + str(today_datetime) + '.log'
log_file = log_name
this_file_name = get_this_file_name()
logger = get_logger(log_file, this_file_name)
logger.info('start')
##############################################################
# rabbit-mq info
##############################################################
conf_path = "conf\hpjy_action_classfiy_rpc.conf"
mq_info = get_mq_info(conf_path)
print(mq_info)

##############################################################
# model
##############################################################
model_name = 'model_self.h5'
model_path = "model/" + model_name
logger.info("load model start")
model = tf.keras.models.load_model(model_path)
image_shape = (28, 28, 1)
logger.info("load model end")

##############################################################
# pika
##############################################################
credentials = pika.PlainCredentials(mq_info['user'], mq_info['password'])
connection = pika.BlockingConnection(pika.ConnectionParameters(mq_info['ip'], mq_info['port'], '/', credentials))
channel = connection.channel()
channel.queue_declare(queue=mq_info['mq_queue_name'], auto_delete=True, exclusive=False, durable=False)

def on_request(ch, method, props, body):
    logger.info("get_imge")
    try:
        strJson = body.decode("utf-8")
        jsonResult = json.loads(strJson)
        strMethod = jsonResult['method']
        image = jsonResult["params"]

        result = imageDecode(image,model)
        strResult = ','.join(str(s) for s in result)
        logger.info(strResult)
        b = strResult.encode(encoding='utf-8')
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id=props.correlation_id),
                         body=b)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        print("error", e)
        logger.error("quit")
        logger.error()
        quit()

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=mq_info['mq_queue_name'], on_message_callback=on_request)
logger.info(" [x] Awaiting RPC requests")
channel.start_consuming()
