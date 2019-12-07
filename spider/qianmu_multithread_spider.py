# encoding: utf-8

"""
@author: sunxianpeng
@file: qianmu_spider.py
@time: 2019/10/26 13:32
"""
import requests
from requests.exceptions import RequestException
from lxml import etree
import threading
from queue import Queue
import time

class Main():
    def __init__(self):
        pass
    def reqest_url(self,url):
        headers = {'user-agent':
                       'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/67.0.3396.62 Safari/537.36'}
        # print("headers = ", headers)
        response = None
        try:
            response = requests.get(url)
            # body = response.text  # 获取网页内容
        except RequestException as e:
            print("request is error!", e)
        return response

    def get_selector(self,html_content):
        selector = None
        try:
            selector = etree.HTML(html_content)
        except Exception as e:
            print("get selector is error!", e)
        return selector

    def analyze_html(self,selector):
        data = {}
        keys = []
        values = []
        data["collage_name"] = selector.xpath('//div[@id="wikiContent"]/h1/text()')[0]
        # 处理单元格内有换行
        table = selector.xpath('//div[@id="wikiContent"]/div[@class="infobox"]//table')
        if table :
            table = table[0]
            cols_k = table.xpath('.//td[1]')
            cols_v = table.xpath('.//td[2]')
            for j in range(len(cols_k)):
                col_k = cols_k[j]
                col_v = cols_v[j]
                keys.append(''.join(col_k.xpath('./p//text()')))
                values.append(''.join(col_v.xpath('./p//text()')))
            # 合并两个列表组成字典,将zip后得到的字典 添加到data字典中
            data.update(zip(keys, values))
        return data

    def download(self,link_queue):
        """ """
        while True:
            # 阻塞，直到从队列取到一个链接
            link = link_queue.get()
            # 取不出链接,或者说取出的是None
            if link is None:
                break
            if not link.startswith("http://www.qianmu.org"):
                link = "http://www.qianmu.org/%s" % link
            try:
                selector = self.get_selector(self.reqest_url(link).text)
                data = self.analyze_html(selector)
                print(data)
            except Exception as e:
                # 此处可以查看相对应的信息，解决表格非标准的形式问题，本次就不处理，直接跳过
                print(link)
                continue
            link_queue.task_done()
            print('remaining queue: %s',link_queue.qsize())


if __name__ == '__main__':
    start_time = time.time()
    m = Main()
    url = "http://www.qianmu.org/ranking/1528.htm"
    link_queue =  Queue()
    req = m.reqest_url(url)
    selector = m.get_selector(m.reqest_url(url).text)
    links = selector.xpath('//tbody//tr[@height=19][position()>1]/td/a/@href')
    for i in range(len(links)):
        link = str(links[i])
        link_queue.put(link)
    # 多线程
    threads = []
    thread_num = 10
    # 启动线程，并将线程对象放入一个列表保存
    for i in range(thread_num):
        t = threading.Thread(target=m.download(link_queue))
        t.start()
        threads.append(t)
    #阻塞队列，直到队列被清空,此时线程未退出
    link_queue.join()
    # 向队列发送n个None，来通知线程退出
    for i in range(thread_num):
        link_queue.put(None)
    # 退出线程
    for t in threads:
        # 堵塞主线程，直到所有的线程退出
        t.join()

    used_time = time.time() - start_time
    print("download finished !!, used time : %s" % used_time)

