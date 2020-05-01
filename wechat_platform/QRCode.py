# encoding: utf-8

"""
@author: sunxianpeng
@file: QRCode.py
@time: 2020/5/1 21:17
"""

import requests
import time
import os
import cv2
import json
import matplotlib.pyplot as plt
class QRCode():
    def __init__(self, common,saveimage, access_token):
        self.common = common
        self.saveimage = saveimage
        self.access_token = access_token

    def __getQRCodeTicket(self, content):
        """
        获取ticket
        :param content:
        :return:
        """
        # 临时二维码 和 永久二维码
        url_ticket = "https://api.weixin.qq.com/cgi-bin/qrcode/create?access_token=" + self.access_token
        # print(url_ticket)
        # expire_seconds 	该二维码有效时间，以秒为单位。 最大不超过2592000（即30天），此字段如果不填，则默认有效期为30秒。
        # action_name 	二维码类型，QR_SCENE为临时的整型参数值，QR_STR_SCENE为临时的字符串参数值，QR_LIMIT_SCENE为永久的整型参数值，QR_LIMIT_STR_SCENE为永久的字符串参数值
        # action_info 	二维码详细信息
        #       scene_id 	场景值ID，临时二维码时为32位非0整型，永久二维码时最大值为100000（目前参数只支持1--100000）
        #       scene_str 	场景值ID（字符串形式的ID），字符串类型，长度限制为1到64
        post_data = json.dumps({"expire_seconds": 604800, "action_name": "QR_SCENE", "action_info": {"scene": {"scene_id": content}}})
        # {"ticket":"gQH47joAAAAAAAAAASxodHRwOi8vd2VpeGluLnFxLmNvbS9xL2taZ2Z3TVRtNzJXV1Brb3ZhYmJJAAIEZ23sUwMEmm3sUw==","expire_seconds":60,"url":"http://weixin.qq.com/q/kZgfwMTm72WWPkovabbI"}
        # ticket 	获取的二维码ticket，凭借此ticket可以在有效时间内换取二维码。
        # expire_seconds 	该二维码有效时间，以秒为单位。 最大不超过2592000（即30天）。
        # url 	二维码图片解析后的地址，开发者可根据该地址自行生成需要的二维码图片
        ticket_json = self.common.requestPOST(url_ticket, post_data).text
        ticket_dict = json.loads(ticket_json)
        ticket = ticket_dict["ticket"]
        return ticket

    def getQRCode(self):
        ticket = self.__getQRCodeTicket(123)
        url_qrcode = "https://mp.weixin.qq.com/cgi-bin/showqrcode?ticket=" + ticket
        print(url_qrcode)
        qrcode = self.common.requestGet(url_qrcode).content
        self.saveimage.LoadImageFromURL("./ qrcode.jpg", qrcode)
        pass

if __name__ == '__main__':
    """获取二维码"""
    pass