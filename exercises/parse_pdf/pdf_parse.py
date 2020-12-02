# encoding: utf-8

"""
@author: sunxianpeng
@file: pdf_parse.py
@time: 2020/12/2 10:44
"""
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed, PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfparser import PDFParser, PDFDocument


def parse(file_read_path):

    pdf_file = open(file_read_path,"rb")
    parser = PDFParser(pdf_file)
    #创建一个PDF文档
    doc = PDFDocument()
    #连接分析器，与文档对象
    parser.set_document(doc)
    doc.set_parser(parser)
    #提供初始化密码，如果没有密码，就创建一个空的字符串
    doc.initialize()
    # 文档是否提供txt转换，不提供就忽略
    if not doc.is_extractable:
        raise PDFTextExtractionNotAllowed
    else:
        # 创建PDF，资源管理器，来共享资源
        rsrcmgr = PDFResourceManager()
        # 创建一个PDF设备对象
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # 创建一个PDF解释其对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # 循环遍历列表，每次处理一个page内容
        # doc.get_pages() 获取page列表
        for page in doc.get_pages():
            interpreter.process_page(page)
            # 接受该页面的LTPage对象
            layout = device.get_result()
            # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象
            # 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等
            # 想要获取文本就获得对象的text属性，
            for x in layout:
                if (isinstance(x, LTTextBoxHorizontal)):
                    with open(r'2.txt', 'a') as f:
                        results = x.get_text()
                        print(results)
                        f.write(results + "\n")


if __name__ == '__main__':
    file_read_path = u"F:\PythonProjects\python_common\exercises\parse_pdf\高中英语词汇词根+联想记忆法  乱序版.pdf"
    parse(file_read_path)