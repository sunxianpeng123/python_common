# encoding: utf-8

"""
@author: sunxianpeng
@file: write_voc_to_txt.py
@time: 2020/6/25 19:41
"""

from data_process.parse_voc import ParsePascalVOC
from configuration import TXT_DIR


if __name__ == '__main__':
    ParsePascalVOC().write_data_to_txt(txt_dir=TXT_DIR)