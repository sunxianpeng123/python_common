# -*- coding: utf-8 -*-
import scrapy


class QianmuCollageRankSpider(scrapy.Spider):
    name = 'qianmu_collage_rank'
    # 允许爬的域名内的url，即爬取的网页必须以改域名开头
    allowed_domains = ['www.qianmu.org']
    # 爬虫的入口地址，可以写多个，
    start_urls = ["http://www.qianmu.org/ranking/1528.htm"]

    def parse(self, response):
        """ start_urls请求成功后，自动调用该方法"""
        # 解析start_urls， extract()提取所有符合条件的文本,extract()返回的永远是列表
        # extract_first()在当解析出来的数据只有一个时可以使用
        links = response.xpath('//tbody//tr[@height=19][position()>1]/td/a/@href')\
            .extract()
        for i in range(len(links)):
            link = str(links[i])
            if not link.startswith("http://www.qianmu.org"):
                link = "http://www.qianmu.org/%s" % link
            try:
                # 让框架即系跟随这个链接，也就是说会再次发起请求
                # 请求成功以后，会调用指定的call_back函数parse_collage
                # 中间过程均是异步调用
                yield response.follow(link,self.parse_collage)
            except Exception as e:
                continue

    def parse_collage(self,response):

        data = {}
        keys = []
        values = []
        data["collage_name"] = response.xpath('//div[@id="wikiContent"]/h1/text()')[0]
        # 处理单元格内有换行
        table = response.xpath('//div[@id="wikiContent"]/div[@class="infobox"]//table')
        if table :
            table = table[0]
            cols_k = table.xpath('.//td[1]')
            cols_v = table.xpath('.//td[2]')
            for j in range(len(cols_k)):
                col_k = cols_k[j]
                col_v = cols_v[j]
                keys.append(''.join(col_k.xpath('./p//text()').extract()))
                values.append(''.join(col_v.xpath('./p//text()').extract()))
            # 合并两个列表组成字典,将zip后得到的字典 添加到data字典中
            data.update(zip(keys, values))
        # yield出去的数据，会被框架接收，进行下一步的处理，
        # 如果没有任何处理，则会打印到控制台
        yield data