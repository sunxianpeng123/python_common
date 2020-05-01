# encoding: utf-8

"""
@author: sunxianpeng
@file: main.py
@time: 2020/5/1 21:10
"""
from AccessToken import *
from Common import *
from QRCode import *
from SaveImage import *

c = Common()
saveimage = SaveImage()
at = AccessToken(c)

"""获取access_token"""
appid = "wxe30c3fab9ed1c379"
appsecret = "1832e564ffbce8bd6d6c8d7e33763f61"
token = at.getAccessToken(appid, appsecret)
# print(token)
"""获取二维码"""
qr = QRCode(c,saveimage, token)
qr.getQRCode()