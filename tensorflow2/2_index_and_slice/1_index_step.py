# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_index_step.py
@time: 2019/12/24 18:59
"""
import tensorflow as tf

def index_start_end_step():
    t = tf.random.normal([4,28,28,3])#4张 大小为28*28*3的图片

    t_1 = t[0:2,:,:,:]
    t_2 = t[:,0:28:2,0:28:2,:]
    t_3 = t[:,:14,:14,:]
    t_4 = t[:,14:,14:,:]
    t_5 = t[:,::2,::2,:]

    print('t.shape = {}'.format(t.shape))
    print('t_1.shape = {}'.format(t_1.shape))
    print('t_2.shape = {}'.format(t_2.shape))
    print('t_3.shape = {}'.format(t_3.shape))
    print('t_4.shape = {}'.format(t_4.shape))
    print('t_5.shape = {}'.format(t_5.shape))
    # t.shape = (4, 28, 28, 3)
    # t_1.shape = (2, 28, 28, 3)
    # t_2.shape = (4, 14, 14, 3)
    # t_3.shape = (4, 14, 14, 3)
    # t_4.shape = (4, 14, 14, 3)
    # t_5.shape = (4, 14, 14, 3)
    return None


# strat:end:step
if __name__ == '__main__':
    index_start_end_step()





