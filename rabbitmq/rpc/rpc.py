#coding=utf-8
import pika
import uuid
import cv2
import numpy as np
import base64
import json
import os
#mq_queue_name='inner.classify.wzry_kill_action'
#mq_queue_name='inner.classify.anchor_detect'
mq_queue_name='inner.classify.anchor'
shape=(832,832)
#queue_ip='120.92.2.84'
queue_ip='58.253.65.94'
#queue_ip='120.92.141.72'
port=5672

class FibonacciRpcClient(object):

    def __init__(self):
        self.credentials = pika.PlainCredentials('admin','admin001')
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
            queue_ip,port,'/',self.credentials))

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



    def getSendJson(self,image,method,uid,id):
        p=[]
        p.append(image)
        sendJson={};
        sendJson['uid']=uid
        sendJson['id']=id
        sendJson['method']=method
        sendJson['params']=p

        return  bytes(json.dumps(sendJson).encode('utf-8'))



    def sendImage(self, image,method):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.sendJson=self.getSendJson(image,method,self.corr_id ,self.corr_id )
        self.channel.basic_publish(
            exchange='',
            routing_key=mq_queue_name,
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=self.sendJson)
        while self.response is None:
            self.connection.process_data_events()

        return self.response



def getKillOrDead(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # canny边缘处理
    edges = cv2.Canny(gray,50,120)
    lines = cv2.HoughLinesP(edges,1,np.pi/360,10,minLineLength=15,maxLineGap=1)
    print(lines)
    if lines is None:
        return False
    else:
        return True

def getActionInfo(img):
    w=img.shape[1]
    m_w=int(w/2)
    img_left=img[:,0:m_w]
    img_right=img[:,m_w:w]
    result={}
    result['kill']=getKillOrDead(img_left)
    result['dead']=getKillOrDead(img_right)


    return result



def findWhiteNum(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([15, 30, 160])
    upper_red = np.array([70, 170, 250])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def showDetectImage(x,y,w,h,img,image_type):

    #img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    x=int(x)
    y=int(y)
    w=int(w)
    h=int(h)
    cv2.rectangle(img,(x,y),(w,h),(0,255,0),3)

    cv2.imshow('result.jpg',img)
    cv2.waitKey(0)

def detectImage(imgfile,fibonacci_rpc):
    image=cv2.imdecode(np.fromfile(imgfile,dtype=np.uint8),cv2.IMREAD_COLOR)
    # img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("11.jpg", img)

    image = cv2.resize(image,shape,interpolation=cv2.INTER_CUBIC)
    h, w =image.shape[:2]
    _,buffer = cv2.imencode('.jpg', image)

    jpg64 = base64.b64encode(buffer)
    jpg_as_text = jpg64.decode("utf-8")

    #response = fibonacci_rpc.call(30)
    response = fibonacci_rpc.sendImage(jpg_as_text,"image_calssify")
    strJson = response.decode("utf-8")
    #print(strJson)
    strJson= eval(strJson)
    print (strJson)
    jsonResult=json.loads(str(strJson))

    # cv2.imshow('result.jpg',image)
    # cv2.waitKey(0)
    # # cv2.imshow('result.jpg',img)
    # # cv2.waitKey(0)
    # for r in  jsonResult:
    #     f_score= r['image_score']
    #     image_type=r['image_type']
    #     # if f_score>0.95:
    #     showDetectImage(r['x'],r['y'],r['w'],r['h'],image,image_type)
    #

#


# for r in  jsonResult:
    #     showDetectImage(r['x'],r['y'],r['w'],r['h'],image)


fibonacci_rpc = FibonacciRpcClient()
#test_path=u"F:\\logdir\\pics\\ddz\\福利样本\\"
#test_path = u"E:\\机器学习样本\\CF识别\\击杀\\"
#test_path =u"E:\\机器学习样本\\CF识别\\生产训练样本\\5kill\\"
test_path=u"E:\\机器学习样本\\和平精英截图\\hpjy\\583276C931B1F2BDCE425BA4F145C7B0\\"
test_path=u"E:\\机器学习样本\\LOL\\yolo_标注\\"
test_path=u"E:\\机器学习样本\\头像\\测试\\"
#test_path=u"E:\\机器学习样本\\LOL\\旧标注\\yolo标注样本\\"
#test_path=u"E:\\英雄联盟截图\\英雄联盟截图\\3\\"
#test_path=u"E:\\机器学习样本\\和平精英截图\\hpjy\\015FF8DBD55917C11B59BC85427FB0BC.gz\\"
#test_path=u"E:\\机器学习样本\\王者荣耀截图\\错误2\\"
#test_path=u"E:\机器学习样本\CF识别\EE5AE62DD810497E9338CF9F26919B82\\"
image_paths = [os.path.join(test_path,i) for i in list(os.listdir(test_path))]
rand=0
fibonacci_rpc.channel
for jpg in image_paths:
    if jpg.endswith(".jpg") :#and rand%33==0:
        print(jpg)
        detectImage(jpg,fibonacci_rpc)

    rand=rand+1

# for i in range(503):
#     jpg=test_path+str(i)+".jpg"
#     detectImage(jpg,fibonacci_rpc)

