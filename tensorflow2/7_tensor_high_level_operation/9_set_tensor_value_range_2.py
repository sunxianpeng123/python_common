# -*- coding: utf-8 -*-
# @Time : 2019/12/31 0:22
# @Author : sxp
# @Email : 
# @File : 9_set_tensor_value_range_1.py
# @Project : python_common

import tensorflow as tf

def tf_clip_by_norm():
    """根据范数对tensor进行裁剪,(等比例放缩)
     tensor中向量会有梯度，梯度具有方向，单纯根据值进行裁剪会影响到向量的梯度和方向，
     根据梯度对向量进行等比例的放缩，可以在不影响梯度等信息的情况下，将tensor值限定在
     一定范围内。
     """
    print("##############1、tf_clip_by_norm #################")
    t = tf.random.normal([2, 2],mean=10)
    t_norm = tf.norm(t)
    # 求出t的L2范数
    print('t_norm = {}'.format(t_norm))#t_norm = 18.598787307739258

    # 先将向量归一化，然后乘以要限定的值，即可以达到将范数限定在一定范围的目的
    t_2 = tf.clip_by_norm(t,15)
    t_2_norm = tf.norm(t_2)
    print('t_2 norm = {}'.format(t_2_norm))#t_2 norm = 15.0

    return None



if __name__ == '__main__':
    tf_clip_by_norm()