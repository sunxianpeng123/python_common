# -*- coding: utf-8 -*-
# @Time : 2020/1/16 8:07
# @Author : sxp
# @Email : 
# @File : two_multi_two.py
# @Project : python_dl

import numpy as np

def get_two_arr():
    res = []
    twos = np.arange(10, 100)
    for num in twos:
        if num % 10 != 0:
            if num not in [11, 12]:
               res.append(num)
    res = np.asarray(res)
    return res

def get_two_multi_two_arr(two_1, two_2):
    res = []
    for t1 in two_1:
        for t2 in two_2:
            mid = str(t1) + ' x ' + str(t2) + ' ='
            # print(mid)
            res.append(mid)
    res = np.array(res)
    return res

def print_in_rules(three_divide_one_arr):
    i = 0
    cols = 5
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
    two_1 = get_two_arr()
    np.random.shuffle(two_1)

    two_2 = get_two_arr()
    np.random.shuffle(two_2)

    two_multi_two_arr = get_two_multi_two_arr(two_1,two_2)
    np.random.shuffle(two_multi_two_arr)
    print_in_rules(two_multi_two_arr)
    print(len(two_multi_two_arr))