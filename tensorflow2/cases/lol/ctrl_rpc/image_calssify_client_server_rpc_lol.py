#coding=utf-8
import pika
import uuid
import numpy as np
import base64
import json
import configparser
import cv2
import time
from skimage import measure


import logging
import os
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

logging.info("start")
cf = configparser.ConfigParser()
a = cf.read("conf/image_calssify_client_server_rpc_lol.conf")
mq_ip = cf.get("rabbit_mq","mq_ip")
mq_port = cf.get("rabbit_mq","mq_port")
mq_user=cf.get("rabbit_mq","mq_user")
mq_pwd=cf.get("rabbit_mq","mq_pwd")
mq_queue_yolo_name=cf.get("rabbit_mq","mq_queue_yolo_name")
mq_queue_callsify_name=cf.get("rabbit_mq","mq_queue_callsify_name")
mq_queue_name=cf.get("rabbit_mq","mq_queue_name")



credentials = pika.PlainCredentials(mq_user,mq_pwd)
connection = pika.BlockingConnection(pika.ConnectionParameters(

    mq_ip,mq_port,'/',credentials))
channel = connection.channel()



channel.queue_declare(queue=mq_queue_name, auto_delete=True,exclusive=False,durable=False)

logging.info(cv2.__version__)

class FibonacciRpcClient(object):

    def __init__(self):
        logging.info("client to server")
        self.credentials = pika.PlainCredentials(mq_user,mq_pwd)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            
            mq_ip,mq_port,'/',self.credentials))

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



    def getSendJson(self,images,method,uid,id):
        sendJson={};
        sendJson['uid']=uid
        sendJson['id']=id
        sendJson['method']=method
        sendJson['params']=images
        strjson=json.dumps(sendJson)
        bjson = strjson.encode(encoding='utf-8')
        return  bjson
    
    def sendToServer(self,image,method,queue):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.sendJson=self.getSendJson(image,method,self.corr_id ,self.corr_id )
        
        self.channel.basic_publish(
            exchange='',
            routing_key=queue,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=self.sendJson)
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
            result= self.sendToServer(image,method,queue)
        except:
            self.__init__()
            result= self.sendToServer(image,method,queue)
        return result




def getKillOrDead(oriImg,img,type,fibonacci_rpc):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # canny边缘处理
    edges = cv2.Canny(gray,50,120)
    lines = cv2.HoughLinesP(edges,1,np.pi/360,8,minLineLength=8,maxLineGap=1)
    logging.info(lines)


    if lines is None :
        return False
    else:
        if len(lines)<=4:
            actionimg=splitImgForDeepClassify(oriImg,type)
            killinfo=killInfo(actionimg,fibonacci_rpc)
            if killinfo=="True":
                return True
            else :
                return False
        else :
            return True

def getActionInfo(oriImg,img,fibonacci_rpc):
    w=img.shape[1]
    m_w=int(w/2)
    img_left=img[:,3:m_w-1]
    img_right=img[:,m_w+2:w]
    result={}
    result['kill']=getKillOrDead(oriImg,img_left,0,fibonacci_rpc)
    result['dead']=getKillOrDead(oriImg,img_right,1,fibonacci_rpc)
    return result



def findWhiteNum(img):
    imgPath=str(uuid.uuid4())+"test.jpg"
    cv2.imwrite(imgPath, img)
    img = cv2.imread(imgPath)
    #os.remove(imgPath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([15, 30, 160])
    # upper_red = np.array([70, 170, 250])
    # lower_red = np.array([20, 30, 160])
    # upper_red = np.array([70, 170, 250])
    lower_red = np.array([12, 35, 160])
    upper_red = np.array([50, 140, 250])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def detectbloodImage(img,fibonacci_rpc):
    logging.info ("start detectbloodImage")
    jpg_as_text=[]
    strResult=""
    try:
        _,buffer = cv2.imencode('.jpg', img)
        jpg64 = base64.b64encode(buffer)
        jpg_as_text.append(jpg64.decode("utf-8"))
    except Exception as e:
        logging.error("imencode_error")
        print("imencode",e)
    response = fibonacci_rpc.sendImage(jpg_as_text,"blood_info_calssify",mq_queue_callsify_name)
    try:
        strResult = response.decode("utf-8")
    except Exception as e:
        logging.error("detectbloodImage_error")
        print("detectbloodImage_error",e)
        strResult=""
    return strResult


def detectkillImage(img,fibonacci_rpc):
    logging.info ("start killImage")
    jpg_as_text=[]
    strResult=""
    try:
        _,buffer = cv2.imencode('.jpg', img)
        jpg64 = base64.b64encode(buffer)
        jpg_as_text.append(jpg64.decode("utf-8"))
    except Exception as e:
        logging.error("imencode_error")
        print("imencode",e)
    response = fibonacci_rpc.sendImage(jpg_as_text,"kill_dead",mq_queue_callsify_name)
    try:
        strResult = response.decode("utf-8")
    except Exception as e:
        logging.error("detectkillImage_error")
        print("detectkillImage_error",e)
        strResult=""
    return strResult


def bloodInfo(cutting_img,fibonacci_rpc):
    y_half=int(cutting_img.shape[0]/2)
    y_half=y_half+int(y_half/6)
    print ("y_half",y_half)
    cutting_img=cutting_img[0:y_half,:]
    result=detectbloodImage(cutting_img,fibonacci_rpc)
    if result=="0":
        return "True"
    else:
        return "False"

def killInfo(cutting_img,fibonacci_rpc):
    result=detectkillImage(cutting_img,fibonacci_rpc)
    if result=="1":
        return "True"
    else:
        return "False"




def splitImage(img):
    w=img.shape[1]
    m_w=int(w/2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([12, 20, 100])
    upper_red = np.array([100, 200, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask=mask)
    img_left=result[:,0:m_w]
    img_right=result[:,m_w:w]
    return img_left,img_right

def  splitImgForDeepClassify(img,type):
    img_left,img_right=splitImage(img)
    if type==0:
        return img_left
    if type==1:
        return img_right
    return

def  classifyByModel(img,fibonacci_rpc):
    img_left,img_right=splitImage(img)
    killinfo=killInfo(img_left,fibonacci_rpc)  #kill
    deadinfo=killInfo(img_right,fibonacci_rpc) #dead
    return (killinfo,deadinfo)


def detectImage(image,fibonacci_rpc):
    logging.info ("start detectImage")
    _,buffer = cv2.imencode('.jpg', image)
    jpg64 = base64.b64encode(buffer)
    jpg_as_text = jpg64.decode("utf-8")
    response = fibonacci_rpc.sendImage([jpg_as_text],"image_calssify",mq_queue_yolo_name)
    try:
        strJson = response.decode("utf-8")
        strJson= eval(strJson)
        logging.info(strJson)
        jsonResult=json.loads(str(strJson))
    except Exception as e:
        print("detectImage_error",e)
        logging.error("detectImage_error")
        jsonResult=[]



    return_result ={}
    return_result['blood']="False"
    return_result['kill']="False"
    return_result['dead']="False"
    for r in  jsonResult:
        type= r['image_type']
        score=r['image_score']
        img=image[r['y']:r['h'],r['x']:r['w']]
        if type=="blood":
           if score >0.9:
               return_result['blood']=bloodInfo(img,fibonacci_rpc)
        elif type=="action_info":
            if score >0.8:
                img=img[1:-1,:]
                action_img  =findWhiteNum(img)
                result =getActionInfo(img,action_img,fibonacci_rpc)
                #kill,dead=classifyByModel(img,fibonacci_rpc)
                # if return_result['kill']=="False":
                #     return_result['kill']=kill
                # if return_result['dead']=="False":
                #     return_result['dead']=dead
                if result['kill']==True:
                     return_result['kill']="True"
                if result['dead']==True:
                     return_result['dead']="True"




    return return_result


fibonacci_rpc = FibonacciRpcClient()
def imageDecode(img_b64encode):
    result=[]
    shape=(832,832)
    try:
        img_b64decode = base64.b64decode(img_b64encode)
        img_array = np.fromstring(img_b64decode,np.uint8)
        img=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
        img = cv2.resize(img,shape)
        result=detectImage(img,fibonacci_rpc)
    except Exception as e:
        print("imageDecode",e)
        logging.error("imageDecode")
    return result



def on_request(ch, method, props, body):
  try :
        logging.info ("-----------------get message----------")
        strJson = body.decode("utf-8")
        jsonResult=json.loads(strJson)
        strMethod=jsonResult['method']
        image=jsonResult["params"][0]

        result=imageDecode(image)
        result=str(result)
        if len(result)>6:
            result=result.replace("'","\"")
        strResult = json.dumps(result)
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
        logging.error("error")
        quit()

        
        

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=mq_queue_name, on_message_callback=on_request)

logging.info(" [x] Awaiting RPC requests")
channel.start_consuming()



