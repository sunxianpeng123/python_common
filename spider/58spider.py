# encoding: utf-8

"""
@author: sunxianpeng
@file: 58spider.py
@time: 2019/10/25 19:19
"""
import requests
from requests.exceptions import RequestException
import numpy as np
import pandas as pd
from lxml import etree

class Main():
    def __init__(self):
        pass

    def reqest_url(self,url):
        headers = {'user-agent':
                       'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/67.0.3396.62 Safari/537.36'}
        # print("headers = ", headers)
        body = ""
        try:
            response = requests.get(url)
            body = response.text  # 获取网页内容
        except RequestException as e:
            print("request is error!", e)
        return body

    def get_url_list(self,html_content):
        # 总的数据条数
        # total = etree.HTML(html_content).xpath('//div[@class="breadFind"]/span/text()')
        items = etree.HTML(html_content).xpath('//h2[@class="clearfix" or @class="clearfix .backcolor"]/a/@href')
        return items

    def analyze_html(self,itemRes):
        # 判断
        def judge(list):
            if list == []:
                return "无"
            else:
                return list[1]

        row = []
        # 公司名称
        companyName = judge(etree.HTML(itemRes).xpath('//span[contains(text(),"全称")]/..//text()'))
        row.append(companyName)
        # 联系人
        linker = judge(etree.HTML(itemRes).xpath('//span[contains(text(),"联系人")]/..//text()'))
        row.append(linker)
        # 电话号码
        telNum = judge(etree.HTML(itemRes).xpath('//span[contains(text(),"电话")]/..//text()'))
        row.append(telNum)
        # 手机号码
        phoneNum = judge(etree.HTML(itemRes).xpath('//span[contains(text(),"手机")]/..//text()'))
        row.append(phoneNum)
        return row

    def save_df_to_excel(self,df, path):
        df.to_excel(path)



if __name__ == '__main__':
    m = Main()
    url = "http://www.likuso.com/city178/p1/"
    html_content = m.reqest_url(url)
    url_list = m.get_url_list(html_content)

    rows = []
    num = 11
    print(len(url_list))
    for i in range(len(url_list)):
        row = []
        if i <= num :
            itemRes = m.reqest_url(url_list[i])
            row = m.analyze_html(itemRes)
            # print(row)
            # 将一条完整的记录插入到最终结果中
            rows.append(row)

    array = np.array(rows)
    columns_name = ["公司名称","联系人","电话号码","手机号码"]
    result_df = pd.DataFrame(array,columns=columns_name)

    save_path = r"F:\PythonProjects\python_study\xiaohulu\spider\data\test.xlsx"
    print(result_df)
    # m.save_df_to_excel(result_df,save_path)

