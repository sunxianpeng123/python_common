# encoding: utf-8

"""
@author: sunxianpeng
@file: settings.py
@time: 2020/6/3 10:48
"""
import configparser


def get_mq_info(conf_path):
    cf = configparser.ConfigParser()
    a = cf.read(conf_path)
    mq_ip = cf.get("rabbit_mq", "mq_ip")
    mq_port = cf.get("rabbit_mq", "mq_port")
    mq_user = cf.get("rabbit_mq", "mq_user")
    mq_pwd = cf.get("rabbit_mq", "mq_pwd")
    mq_queue_yolo_name = cf.get("rabbit_mq", "mq_queue_yolo_name")
    mq_queue_callsify_name = cf.get("rabbit_mq", "mq_queue_callsify_name")
    mq_queue_name = cf.get("rabbit_mq", "mq_queue_name")
    mq_info = {
        "ip": mq_ip,
        "port": mq_port,
        "user": mq_user,
        "password": mq_pwd,
        "mq_queue_yolo_name": mq_queue_yolo_name,
        "mq_queue_callsify_name": mq_queue_callsify_name,
        "mq_queue_name": mq_queue_name
    }
    return mq_info
