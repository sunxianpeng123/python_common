FROM python:3.6-slim

RUN pip3 install opencv-python==4.1.1.26   numpy pika scikit_image --index-url=https://mirrors.aliyun.com/pypi/simple



RUN apt-get update &&  apt-get install -y libsm6 libxrender1 libxext-dev 
RUN apt-get -y install libglib2.0-dev

WORKDIR /darknet

COPY . /darknet
    

ENTRYPOINT [ "python", "server.py"]
 
