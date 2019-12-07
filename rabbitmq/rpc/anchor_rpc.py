# encoding: utf-8

"""
@author: sunxianpeng
@file: rpc_client.py
@time: 2019/11/27 11:32
"""

import pika
import uuid
import time
import cv2
import base64
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
class FibonacciRpcClient(object):
    def __init__(self,mq_user,mq_pwd,mq_ip,mq_port):
        logging.info("client to server")
        # 添加用户名和密码
        self.credentials = pika.PlainCredentials(mq_user, mq_pwd)
        # 配置连接参数
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            mq_ip, mq_port, '/', self.credentials))
        # 创建一个信道
        self.channel = self.connection.channel()
        # 声明一个队列,durable参数声明队列持久化,把消费者和queue绑定起来
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.__on_response,
            auto_ack=True)

    def __on_response(self,ch,method,props,body):
        # 如果收到的ID和本机生成的相同，则返回的结果就是我想要的指令返回的结果
        if self.corr_id == props.correlation_id:
            self.response = body

    def __getSendJson(self,images,method,uid,id):
        print("get send json ...")
        sendJson={}
        sendJson['uid']=uid
        sendJson['id']=id
        sendJson['method']=method
        sendJson['params']=images
        strjson=json.dumps(sendJson)
        bjson = strjson.encode(encoding='utf-8')
        return bjson

    def __sendToServer(self,image,method,queue):
        print("send to server ...")
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.__sendJson=self.__getSendJson(image,method,self.corr_id ,self.corr_id )
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=self.__sendJson)
        i=0
        while self.response is None:
            i=i+1
            self.connection.process_data_events()
            if i>3000:
                logging.info ("time out")
                break
            time.sleep(0.01)
        return self.response

    def sendImage(self, image,method,queue):
        try:
            print("send to image ...")
            result= self.__sendToServer(image,method,queue)
        except:
            self.__init__()
            result= self.__sendToServer(image,method,queue)
        return result

def image_to_base64(path):
    """将cv2读取的图片转成算法需要的格式"""
    img = cv2.imread(path)
    _,buffer = cv2.imencode('.jpg', img)
    jpg64 = base64.b64encode(buffer)
    jpg_as_text = jpg64.decode("utf-8")
    return img,jpg_as_text

def base64_to_image(base64_code):
    """将base64格式的图片转成bgr格式"""
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
    img = img[:, :, (2, 1, 0)]
    return img

def plt_face_dot(img,data):
    """将识别出的脸部信息在原图上画出"""
    result = json.loads(json.loads(data))
    for k,v in result.items():
        x,y = int(v.split(',')[0]),int(v.split(',')[1])
        # print("k = {},x = {},y = {}".format(k,x,y))
        if x != -1 and y != -1:
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    img = img[:,:, (2, 1, 0)]
    plt.imshow(img)
    plt.show()

def plot_person_location(img,data):
    """将识别出的人物框图画出"""
    persons_location_list = json.loads(json.loads(data))
    for one_location in persons_location_list:
        x = None;y = None;w = None;h = None
        for k, v in one_location.items():
            if k == 'image_type':
                print("image_type = {}".format(v))
            elif k == 'image_score':
                print("image_score = {}".format(v))
            else:
                if k == 'x':
                    x = v
                if k == 'y':
                    y = abs(v)
                if k == 'w':
                    w = v
                if k == 'h':
                    h = v
         # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
        left_coner_x = x;left_coner_y = y;right_coner_x = w;right_coner_y = h
        print('左上角坐标 ：x = {},y = {},右下角坐标 ：x = {},y = {}'.format(left_coner_x, left_coner_y,right_coner_x, right_coner_y))
        cv2.rectangle(img, (left_coner_x, left_coner_y), (right_coner_x, right_coner_y), (0, 255, 0), 4)
        # 截取出识别的人物
        # img_person = img[y:h, x:w]
        # img_person = img_person[:, :, (2, 1, 0)]
        img = img[:, :, (2, 1, 0)]
        plt.imshow(img)
        plt.show()

def get_located_person(original_img,data):
    """获取识别出的人物框图"""
    persons_location_list = json.loads(json.loads(data))
    persons = []
    for one_location in persons_location_list:
        x = None;y = None;w = None;h = None
        for k, v in one_location.items():
            if k == 'image_type':
                print("image_type = {}".format(v))
            elif k == 'image_score':
                print("image_score = {}".format(v))
            else:
                if k == 'x':
                    x = v
                if k == 'y':
                    y = abs(v)
                if k == 'w':
                    w = v
                if k == 'h':
                    h = v
        # 截取出识别的人物
        img_person = original_img[y:h, x:w]
        persons.append(img_person)
    return persons

if __name__ == '__main__':
    image_path = r'F:\PythonProjects\python_study\rabbitmq\rpc\data\test7.jpg'
    mq_ip = '120.92.2.84'
    mq_port = 5672
    mq_user = 'admin'
    mq_pwd = 'admin'
    mq_queue_name = 'inner.classify.anchor_sxp'
    mq_queue_yolo_name = 'inner.classify.anchor_detect_sxp'
    mq_queue_callsify_name = 'inner.classify.anchor_body_action_sxp'
    used_mq_queue = mq_queue_name
    #
    fibonacci_rpc = FibonacciRpcClient(mq_user,mq_pwd,mq_ip,mq_port)
    img,jpg_as_text = image_to_base64(path=image_path)
    response = fibonacci_rpc.sendImage([jpg_as_text],"sxp_control",used_mq_queue)
    print(response)

    if used_mq_queue == mq_queue_yolo_name:
        plot_person_location(img,response)
    if used_mq_queue == mq_queue_callsify_name:
        plt_face_dot(img, response)
    if used_mq_queue == mq_queue_name:
        location_response = fibonacci_rpc.sendImage([jpg_as_text],'yolov3_sxp',mq_queue_yolo_name)
        # print(location_response)
        person = get_located_person(img,location_response)[0]
        plt_face_dot(person, response)
