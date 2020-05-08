# -*- coding: utf-8 -*-
# @Time : 2020/1/16 8:50
# @Author : sxp
# @Email : 
# @File : shuffle_compute.py
# @Project : python_dl

import numpy as np

def get_two_minus_two_tup_list(two_1, two_2):
    res = []
    two_1 = [x for x in two_1 if x % 10 != 0]
    np.random.shuffle(two_1)
    two_2 = [x for x in two_2 if x % 10 != 0]
    np.random.shuffle(two_2)

    for num1 in two_1:
        for num2 in two_2:
            diff = num1 - num2
            if diff >= 3 and diff % 10 != 0 and diff <=9:
                tup = (num1, num2)
                res.append(tup)
    return res

def get_two_add_two_list(two_1, two_2):
    res = []
    for num1 in two_1:
        for num2 in two_2:
            add = num1 + num2
            if add < 100 and add % 10 != 0:
                tup = (num1, num2)
                res.append(tup)
    return res


def get_three_divid_two_minus_two_list(threes, two_minus_two_list):
    res = []
    for three in threes:
        for tup in two_minus_two_list:
            num1 = tup[0]
            num2 = tup[1]
            diff = num1 - num2
            if three % diff == 0 and three % 10 != 0 and three % 100 !=0:
                str_math = str(three) + ' ÷ (' + str(num1) + '-' + str(num2) + ') = '
                # print(str_math)
                res.append(str_math)
    return res

def get_two_multi_two_add_two_list(twos, two_add_two_list):
    res = []
    twos = [x for x in twos if x % 10 != 0]
    for two in twos :
        for tup in two_add_two_list:
            num1 = tup[0]
            num2 = tup[1]
            str_math = str(two) + ' x (' + str(num1) + ' + ' + str(num2) +') = '
            res.append(str_math)
    return res



def print_in_rules(three_divide_one_arr):
    i = 0
    cols = 3
    mid = ''
    for one_math in three_divide_one_arr:
        if i <= cols:
            if mid == '':
                mid = one_math
            else:
                mid = mid + '    ' + one_math
            i += 1
        else:
            print(mid + '  ')
            print()
            mid = ''
            i = 0


if __name__ == '__main__':
    two_1 = np.arange(10,100)
    np.random.shuffle(two_1)

    two_2 = np.arange(10,100)
    np.random.shuffle(two_2)

    threes =[x for x in np.arange(100, 1000) if x % 10 != 0 and x % 100 != 0]
    np.random.shuffle(threes)

    # 两位数相减
    two_minus_two_list = get_two_minus_two_tup_list(two_1, two_2)
    # 两位数相加
    two_add_two_list = get_two_add_two_list(two_1, two_2)
    # 三位数除以（两位数相减）
    three_divid_two_minus_two_list = get_three_divid_two_minus_two_list(threes,two_minus_two_list)
    # 两位数乘以（两位数相加）
    two_multi_two_add_two_list = get_two_multi_two_add_two_list(two_1, two_add_two_list)
    # 合并结果成list
    result = three_divid_two_minus_two_list + two_multi_two_add_two_list
    # 洗牌
    result = np.asarray(result)
    np.random.shuffle(result)
    print_in_rules(result)
    print(len(result))