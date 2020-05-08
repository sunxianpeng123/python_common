# -*- coding: utf-8 -*-
# @Time : 2020/1/15 15:38
# @Author : sxp
# @Email : 
# @File : three_divide_one.py
# @Project : python_dl

import numpy as np

def get_three_divide_arr(ones, threes):
    res = []
    for one in ones:
        for three in threes:
            mid = str(three) + ' ÷ ' + str(one) + ' ='
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
    threes = np.arange(100,1000)
    np.random.shuffle(threes)
    # print(threes[:5])

    ones = np.arange(3, 10)
    np.random.shuffle(ones)
    # print(ones)

    three_divide_one_arr = get_three_divide_arr(ones, threes)
    np.random.shuffle(three_divide_one_arr)
    print_in_rules(three_divide_one_arr)




