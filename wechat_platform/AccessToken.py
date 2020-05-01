# encoding: utf-8

"""
@author: sunxianpeng
@file: access_token.py
@time: 2020/5/1 18:33
"""
import requests
import json
import os
import time
class AccessToken():
    def __init__(self,common):
        self.common = common

    def getFileTime(self,filepath):
        """
        获取文件时间相关信息
        :param filepath:
        :return:
        """
        # format = '%Y-%m-%d %H:%M:%S'
        # t = time.ctime(os.path.getmtime(filepath))  # 文件最后的更新时间
        t = time.ctime(os.path.getctime(filepath))  # 文件创建时间
        timestamp = time.mktime(time.strptime(t))
        return timestamp

    def __getTokenFromFile(self,filepath):
        """
        tocken有效，则从文件中获取
        :param filepath:
        :return:
        """
        content = ""
        try:
            fp = open(filepath)
            content = fp.read()
            fp.close()
        except IOError:
            print("ERROR!! 从文件中读取token失败")
        return content

    def __writeTokenToFile(self,filepath, token):
        """
        将新获取的token写入file
        :param filepath:
        :return:
        """
        success = False
        try:

            fp = open(filepath,'w')
            fp.write(token)
            fp.close()
            success = True
        except IOError:
            print("ERROR!! 文件写入token失败")
        return success

    def getAccessToken(self, appid, appsecret,token_file='./access_token'):
        """
        获取access_token
        需要考虑token的过期时间问题，不过期则存储在文件中，过期则重新生成
        :param appid:
        :param appsecret:
        :return:
        """
        token = ""
        limit_time = 7200
        now_timestamp = time.time()
        # 当前时间减去文件最后更新时间即token已经存在的时间,即token有效
        if(os.path.exists(token_file) and (now_timestamp - self.getFileTime(token_file) < limit_time)):
            # print("time.time() = {}".format(time.time()))
            token_json = self.__getTokenFromFile(token_file)
        else:
            url = "https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid="+ appid+"&secret="+ appsecret
            # print(url)
            token_json = self.common.requestGet(url).text
            self.__writeTokenToFile(token_file,token_json)

        try:
            token_dict = json.loads(token_json)
            token = token_dict["access_token"]
            # print(type(token_dict))
            # for kv in token_dict.items():
            #     print(kv[0]+"__"+str(kv[1]))
        except ValueError:
            print("ERROR!! 解析get请求返回的json错误")
        return token


if __name__ == '__main__':
    at = AccessToken()
    """获取access_token"""
    appid = "wxe30c3fab9ed1c379"
    appsecret = "1832e564ffbce8bd6d6c8d7e33763f61"
    at.getAccessToken(appid, appsecret)



