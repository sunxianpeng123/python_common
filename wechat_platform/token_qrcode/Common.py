# encoding: utf-8

"""
@author: sunxianpeng
@file: common.py
@time: 2020/5/1 21:12
"""

import requests
import json
class Common():
    def __init__(self):
        pass
    def requestGet(self,url,ssl=True):
        """
        发送 GET 请求
        :param url:
        :param ssl: boolean，是否为 https 协议,这个后面再去查怎么做
        :return: string，响应主题 content,{"access_token":"ACCESS_TOKEN","expires_in":7200}
        """
        # 请求代理信息
        user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:75.0) Gecko/20100101 Firefox/75.0"
        header = {"User-Agent": user_agent}
        r = requests.get(url,headers=header)
        print(r.status_code, r.reason)
        # r = json.dumps({"access_token":"ACCESS_TOKEN","expires_in":7200})
        return r

    def requestPOST(self,url,post_data,ssl=True):
        """
        发送 POST 请求
        :param url:
        :param post_data:
        :param ssl:
        :return:

        """
        user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:75.0) Gecko/20100101 Firefox/75.0"
        header = {"User-Agent": user_agent,
               "Accept-Ranges":"bytes",
                "Cache-control":"max-age=604800",
                "Connection":"keep-alive",
                "Content-Length":"28026",
                "Content-Type":"image/jpg",
                "Date":"Wed, 16 Oct 2013 06:37:10 GMT" ,
                "Expires":"Wed, 23 Oct 2013 14:37:10 +0800" ,
                "Server":"nginx/1.4.1"
        }

        r = requests.post(url , data=post_data, headers=header)

        return  r


if __name__ == '__main__':
    pass