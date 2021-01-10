# encoding: utf-8

"""
@author: sunxianpeng
@file: handle_parquet.py
@time: 2021/1/6 19:40
"""
import os
import shutil


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    read_dir = r"C:\Users\Dell\Desktop\repair\test"
    write_dir = r"C:\Users\Dell\Desktop\repair\12-13"

    filenames = os.listdir(read_dir)
    for filename in filenames:
        print(len(filename) - 16)
        filepath = os.path.join(read_dir, filename)
        name = filename[0:67]
        time = filename[67:len(filename)]
        hour = int(time.split("-")[2].split("_")[0])
        minute = int(time.split("-")[2].split("_")[1])
        second = int(time.split("-")[2].split("_")[2])

        if minute >= 12 and minute < 13:
            writepath = os.path.join(write_dir, name)
            print("filepath = {}".format(filepath))
            print("writepath = {}".format(writepath))
            print("name = {},,time = {}".format(name, time))
            print("hour = {},,minute = {},, second = {}".format(hour, minute, second))
            shutil.move(filepath, writepath)
        # elif minute == 12 and second <= 28:
        #     writepath = os.path.join(write_dir, name)
        #     print("filepath = {}".format(filepath))
        #     print("writepath = {}".format(writepath))
        #     print("name = {},,time = {}".format(name, time))
        #     print("hour = {},,minute = {},, second = {}".format(hour, minute, second))
        #     shutil.move(filepath, writepath)
        else:
            continue
