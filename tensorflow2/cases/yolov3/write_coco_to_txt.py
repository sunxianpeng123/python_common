# encoding: utf-8

"""
@author: sunxianpeng
@file: write_coco_to_txt.py
@time: 2020/6/25 19:40
"""

from data_process.parse_coco import ParseCOCO
from configuration import TXT_DIR


if __name__ == '__main__':
    coco = ParseCOCO()
    coco.write_data_to_txt(txt_dir=TXT_DIR)