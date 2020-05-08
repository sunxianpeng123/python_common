# -*- coding: utf-8 -*-
# @Time : 2020/2/3 9:42
# @Author : sxp
# @Email : 
# @File : english_code.py
# @Project : python_common

import os
import re

def get_all_lines_list(txt_path):
    with open(txt_path,'r',encoding='utf-8') as f:
        res = f.readlines()
        # print(lines)
    return res

def get_distincted_eng_code(chars, lines):
    tSet = set()
    for line in lines:
        eng_code = ''
        line = line.strip()
        if len(line) != 0 and line != '……' and line not in chars:
            line = re.sub(' +', ' ', line)
            # print(line)
            kv_list = line.split(':')
            # print(len(kv_list))
            if len(kv_list) == 2:
                english = kv_list[0].strip()
                code = kv_list[1].strip()
                eng_code = english + ' : ' + code
                # print('english = {} , code = {}'.format(english,code))
            if len(kv_list) == 1:
                kv_list = line.split(':')
                if len(kv_list) == 2:
                    english = kv_list[0].strip()
                    code = kv_list[1].strip()
                    eng_code = english + ' : ' + code
                    # print(kv_list)
                    # print('english = {} , code = {}'.format(english,code))
                if len(kv_list) == 1:
                    kv_list = line.split('：')
                    if len(kv_list) == 2:
                        english = kv_list[0].strip()
                        code = kv_list[1].strip()
                        eng_code = english + ' : ' + code
                        # print(kv_list)
                        # print('english = {} , code = {}'.format(english,code))
                    if len(kv_list) == 1:
                        kv_list = line.replace('）', '').split('（')
                        if len(kv_list) == 2:
                            english = kv_list[0].strip()
                            code = kv_list[1].strip()
                            eng_code = english + ' : ' + code
                            # print(kv_list)
                            # print('english = {} , code = {}'.format(english,code))
        eng_code = eng_code.replace('(', '').replace(')', '').replace('（', '').replace('）', '')
        tSet.add(eng_code)
    return tSet

def get_ordered_by_start_char(tSet):
    tMap = {}
    for eng_code in tSet:
        if eng_code != '':
            k = eng_code[0].upper()
            if k in tMap.keys():
                l = tMap.get(k)
                l.append(eng_code)
                tMap[k] = l
            else:
                v_list = []
                v_list.append(eng_code)
                tMap[k] = v_list
    return tMap

def write_result(tMap,write_path):

    with open(write_path, 'w',encoding='utf-8') as f:
        for tup in tMap:
            type = tup[0]
            eng_code_list = tup[1]
            print(eng_code_list)
            f.write(type)
            f.write('\n')
            for eng_code in eng_code_list:
                f.write(eng_code)
                f.write('\n')


if __name__ == '__main__':
    txt_path = r'E:\hdjy\英语二级代码库.txt'
    write_path = r'E:\PythonProjects\python_common\HDJY\memory\english_code.txt'
    lines = get_all_lines_list(txt_path)
    chars = ['a','b','c','d', 'e', 'f', 'g', 'h', 'i', 'j','k', 'l','m', 'n','o','p','q','r','s','t','u','v','w','x','y','z',
             'A','B','C','D', 'E', 'F', 'G', 'H', 'I', 'J','K', 'L','M', 'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    tSet = get_distincted_eng_code(chars, lines)
    tMap = get_ordered_by_start_char(tSet)

    tMap = sorted(tMap.items(),key=lambda d:d[0],reverse=False)
    print(tMap)
    # tList.sort(key=lambda x:x[0])
    """结果存储"""
    write_result(tMap, write_path)

