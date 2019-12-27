# -*- coding: utf-8 -*-
# @Time : 2019/12/28 1:16
# @Author : sxp
# @Email : 
# @File : for_test.py
# @Project : python_common

import tensorflow as tf

def test():
    x = tf.ones([4,2])
    W = tf.ones([2,1])
    b = tf.constant(0.1)

    y = x@W + b
    print(y)
    print("==========================")

    out = tf.nn.relu(y)
    print(out)


if __name__ == '__main__':
    test()