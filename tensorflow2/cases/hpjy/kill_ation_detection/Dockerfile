FROM tensorflow/tensorflow:2.0.0b1-py3


RUN pip3 install opencv-python  pika  scikit_image --index-url=https://mirrors.aliyun.com/pypi/simple

RUN apt-get install -y libsm6 libxrender1 libxext-dev \
    && apt-get autoremove

WORKDIR /darknet

COPY . /darknet    

ENTRYPOINT [ "python", "hpjy_action_classfiy.py"]
 
