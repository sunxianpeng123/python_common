#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pika
import tensorflow as tf
import cv2
import numpy as np
import base64
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from skimage import measure
import json
import configparser
import uuid

import logging
import os
import time
# 第一步，创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
# 第二步，创建一个handler，用于写入日志文件
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path ='Logs/'
log_name = log_path  + 'logs.log'
logfile = log_name
fh = logging.FileHandler(logfile, mode='w')
ch = logging.StreamHandler()
fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
ch.setLevel(logging.INFO)
# 第三步，定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# 第四步，将logger添加到handler里面
logger.addHandler(fh)
logger.addHandler(ch)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = whit_gpu
model_name='model_self.h5'
cf = configparser.ConfigParser()
a = cf.read("hpjy_action_classfiy_rpc.conf")
mq_ip = cf.get("rabbit_mq","mq_ip")
mq_port = cf.get("rabbit_mq","mq_port")
mq_user=cf.get("rabbit_mq","mq_user")
mq_pwd=cf.get("rabbit_mq","mq_pwd")
mq_queue_name=cf.get("rabbit_mq","mq_queue_name")


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

credentials = pika.PlainCredentials(mq_user,mq_pwd)
connection = pika.BlockingConnection(pika.ConnectionParameters(
    mq_ip,mq_port,'/',credentials))
channel = connection.channel()
channel.queue_declare(queue=mq_queue_name, auto_delete=True,exclusive=False,durable=False)



# def lap_filter2(imgs, k_max=6):
#     result=[]
#     for i in imgs:
#
#         img = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
#         if np.sum(img) < 0.1:
#             result.append(f1)
#         else:
#             kernel_lap = np.array([[0, -1, 0],
#                                    [-1, k_max, -1],
#                                    [0, -1, 0]])
#             f1 = cv2.filter2D(img, -1, kernel_lap)
#             fz=200
#             f1[f1 < fz] = 0
#             f1[f1 >= fz] = 1
#             result.append(f1)
#     return result
#
#



def classify_gray(model,imgs):
    image_shape = (28, 28, 1)
    predict_image_set=[]
    for i in imgs:
        img=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,image_shape[:2])
        data = np.array(img).reshape(image_shape[0],image_shape[1],1)
        predict_image_set.append(data)

    pd=np.array(predict_image_set)
    rs = model.predict(pd)
    result = np.argmax(rs, axis=1)
    #print (rs,result)
    r_no=0
    returnList=[]
    for r in result :
        score=rs[r_no][r]
        logging.info(str(r)+"得分:"+str(score))
        #print ("得分:",score,r)
        if score<0.8:
            returnList.append(-1)
        else:
            returnList.append(r)
        r_no=r_no+1

    return returnList




model_path=model_name
logging.info("start")
model1 = tf.keras.models.load_model(model_path)
image_shape = (28,28, 1)
logging.info("end")



def kill_classify(images):
    result=[]
    try:
        result =classify_gray(model1,images)
        saveImage(result,images)
    except:
        logging.error ("error image")
        result=[]
         
    return result



def imageDecode(img_b64encode):
  r=[]
  try:
    images=[]
    for img64 in img_b64encode:
        img_b64decode = base64.b64decode(img64)
        img_array = np.fromstring(img_b64decode,np.uint8) 
        img=cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
        images.append(img)
    r=kill_classify(images)
  except Exception as e:
       print("error",e)
       logging.error("imageDecodee_error")
  return r

def saveImage(result,images):
    logging.info ("save")
    i=0
    path="img_test"
    if not os.path.exists(path):
         os.mkdir(path)
    for img in  images:
        num=str(result[i])
        
        imgdir=path+"//"+num
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)
        imageID=str(uuid.uuid1())+".jpg"
        imgPath=imgdir+"//"+imageID
        
        cv2.imwrite(imgPath, img)
        i=i+1
            
def on_request(ch, method, props, body):
     logging.info ("get_imge")
     try:
        strJson = body.decode("utf-8")
        jsonResult=json.loads(strJson)
        strMethod=jsonResult['method']
        image=jsonResult["params"]
        
        result=imageDecode(image)
        strResult = ','.join(str(s) for s in result)
        logging.info(strResult)
        b = strResult.encode(encoding='utf-8')
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id = \
                                                             props.correlation_id),
                         body=b)
        ch.basic_ack(delivery_tag=method.delivery_tag)
     except Exception as e:
         print("error",e)
         logging.error("quit")
         logging.error()
         quit()
        

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=mq_queue_name, on_message_callback=on_request)

logging.info(" [x] Awaiting RPC requests")
channel.start_consuming()

    

