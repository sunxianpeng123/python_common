# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_indoor.py
@time: 2019/10/26 18:37
"""
import scrapy

class Main(scrapy.Spider):
    def __init__(self):
        pass
    # 爬虫的名字
    name ="quote"
    # 起始的url列表
    start_urls = ['http://quotes.toscrape.com/']

    # 固定函数，固定写法
    def parse(self, response):
        # css和 xpath 都可以实现
        # quotes = response.css("div.quote")
        quotes = response.xpath('//div[@class="quote"]')#和上述作用一样
        for quote in quotes:
            yield {
                # extrack_first提取第一个选择器的内容cd qian
                "text": quote.css('span.text::text').extract_first(),
                "author": quote.xpath('./span/small/text').extract_first()
            }
        next_page = response.xpath('//li[@class="next"]/a/@href').extract_first()

        if next_page:
            # page下载完后，交给谁处理(再交给parse函数处理)
            yield response.follow(next_page,self.parse)



if __name__ == '__main__':
    m = Main()

