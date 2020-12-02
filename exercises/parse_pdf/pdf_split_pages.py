# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2020/12/1 16:31
"""
import os

from PyPDF2 import PdfFileReader, PdfFileWriter




# def delete_pdf(index,file_path):
#     pages = file_path.getNumPages() // 3
#     for i in range(pages):
#         if i + 1 in index:
#             continue
#         output.addPage(file_path.getPage(i)) // 4
#
#     outputStream = open("PyPDF2-output.pdf", "wb")
#     output.write(outputStream) // 5

if __name__ == '__main__':
    file_read_path = u"F:\PythonProjects\python_common\exercises\parse_pdf\高中词汇英译中练习.pdf"
    file_read_basename = os.path.basename(file_read_path)
    file_read_name = file_read_basename.split(".")[0]
    print("file_read_basename = {}, file_read_name = {}".format(file_read_basename,file_read_name))
    # read
    pdf_read_file = open(file_read_path, 'rb')
    pdf_reader = PdfFileReader(pdf_read_file,strict=False)  # 将要分割的PDF内容格式话
    pages = pdf_reader.getNumPages()
    print("pdf pages number = {}".format(pages))
    # write
    pdf_writer = PdfFileWriter()
    file_write_dir = u"F:\PythonProjects\python_common\exercises\parse_pdf"
    threold = 2
    split_page_num = 0
    for i in range(pages):
        print("this is the {} page".format(i))
        if split_page_num == threold:
            file_write_basename = file_read_basename + "_" + str(i-1) + "-" + str(i) + ".pdf"
            # file_write_basename = os.path.join(file_write_name)
            print(file_write_basename)
            file_write_path = os.path.join(file_write_dir,file_write_basename)
            pdf_write_file = open(file_write_path, "wb")
            pdf_writer.write(pdf_write_file)
            pdf_writer = PdfFileWriter()
            pdf_writer.addPage(pdf_reader.getPage(i))
            split_page_num = 1
        else:
            pdf_writer.addPage(pdf_reader.getPage(i))
            split_page_num += 1
    #兜底
    file_write_basename = file_read_basename + "_" + "last"+ ".pdf"
    # file_write_basename = os.path.join(file_write_name)
    print(file_write_basename)
    file_write_path = os.path.join(file_write_dir, file_write_basename)
    pdf_write_file = open(file_write_path, "wb")
    pdf_writer.write(pdf_write_file)


