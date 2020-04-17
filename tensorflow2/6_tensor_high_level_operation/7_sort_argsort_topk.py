# encoding: utf-8

"""
@author: sunxianpeng
@file: t_sort_argsort_topk.py
@time: 2019/12/30 16:20
"""

import tensorflow as tf

def tf_sort_argsort():
    """
    sort:将tensor排序
    argsort：返回排序后的元素在原tensor中的下标
    """
    print("###################1、tf_sort_argsort ####################")
    print("一维tensor=================")
    a = tf.random.shuffle(tf.range(5))
    # DESCENDING 降序,ASCENDING升序
    b = tf.sort(a,direction='DESCENDING')
    # 返回排序后的元素在原tensor中的下标
    idx = tf.argsort(a,direction='DESCENDING')
    # 使用gather获取排序后的数据
    b_1 = tf.gather(a,idx)

    print('a = {}'.format(a))
    print('b = {}'.format(b))
    print('idx = {}'.format(idx))
    print('b_1 = {}'.format(b_1))
    # a = [2 0 3 4 1]
    # b = [4 3 2 1 0]
    # idx = [3 2 0 4 1]
    # b_1 = [4 3 2 1 0]
    print("二维tensor=================")
    c = tf.random.uniform([2,3],maxval=10,dtype=tf.int32)
    # 默认axis为 最后一个维度，此处即axis=1，表示按照指定的axis上的shape进行分组，每组数据为3个（即每行三个数据）
    d = tf.sort(c)
    # 指定axis=0，表示按照第一个维度，每组数据有两个，在每组数据中排序
    e = tf.sort(c,axis=0)
    # DESCENDING 降序,ASCENDING升序
    f = tf.sort(c,direction='DESCENDING')
    #默认axis为最后一个维度，返回指定维度排序后的元素在原tensor中的下标
    idx = tf.argsort(c)
    #
    print('c = {}'.format(c))
    print('d = {}'.format(d))
    print('e = {}'.format(e))
    print('f = {}'.format(f))
    print('idx = {}'.format(idx))
    # c = [[4 1 7]
    #  [1 0 7]]

    # d = [[1 4 7]
    #  [0 1 7]]

    # e = [[1 0 7]
    #  [4 1 7]]

    # f = [[7 4 1]
    #  [7 1 0]]

    # idx = [[1 0 2]
    #  [1 0 2]]
    return None

def tf_topk():
    """返回最内层一维（也就是最后一维）的前k个最大的元素，以及它所对应的索引。
    返回的值除了最后一维维度为k之外，其它维度维持原样。"""
    print("###################2、tf_topk ####################")
    a = tf.random.uniform([2,5],maxval=10,dtype=tf.int32)
    # 默认为最后一个维度,此处即axis=1，即按照 axis 的shape进行分组（每组数据个数为shape个），
    # 返回每组数据中的排序后的前两个数据，和前两个数据在原tensor中的下标索引
    #注意： sorted:如果为真，则得到的k个元素将按降序排列。False则表示不会将得到的topk个元素排序
    b = tf.math.top_k(a, k=3, sorted=False)#true 表示降序，false表示 不排序
    print('a = {}'.format(a))
    print('b indeices = {}'.format(b.indices))
    print('b values = {}'.format(b.values))
    # a = [[4 0 8]
    #  [3 6 6]]

    # b indeices = [[2 0]
    #  [1 2]]

    # b values = [[8 4]
    #  [6 6]]
    return None

def tf_topk_accuracy():
    k = 3#
    data_size = 2
    """ 用测结果的前k个标签来衡量模型的准确率"""
    print("###################3、tf_topk_accuracy ####################")
    # 预测值
    preds = tf.constant([[0.1, 0.2, 0.7],
                        [0.2, 0.7, 0.1]])
    # 真实值，所有样本的真实标签
    y = tf.constant([2, 0])
    # 将shape为（2，）的tensor转为 shape=（3,2）列的tensor，方便与top_k_indices进行比较, y_表示在 y变量上修改数据
    # 列表示样本的真实值
    y_ = tf.broadcast_to(y,[k,data_size])

    # 取出前k个最大概率的topk信息，并按照降序排序
    top_k = tf.math.top_k(preds, k, sorted=True)
    # 取出前k个概率最大的概率在原tensor中的下标索引，本例中下标索引就是预测的标签
    top_k_indices = top_k.indices
    # 指定 top_k_indices 的维度顺序，即 perm 指的是top_k_indices中维度的下标
    # 在进行转置之前，top_k_indices的每行表示的是 对同一个样本预测的标签，
    # 在进行转置之后，top_k_indices的每行表示的是对所有样本的预测标签，刚好和y相对应
    top_k_indices = tf.transpose(top_k_indices,perm=[1,0])
    # 比较预测值和真实值，得到boolean矩阵
    correct = tf.equal(top_k_indices,y_)#3*2
    print(correct.shape)

    res = []
    for i in range(1,k + 1):
        # 取出前index行，并将前k行tensor进行转置，-1表示系统 自动求解转置 的形状大小
        t = tf.reshape(correct[:i], [-1])
        # 将 boolean 类型转成 float 类型，即转成 1.0 和 0.0
        correct_i = tf.cast(t,dtype=tf.float32)
        # 求解预测正确 的个数
        correct_i = tf.reduce_sum(correct_i)
        acc = float(correct_i / data_size )
        # 将top1 ,top2, top3 的准确率追加到res中
        res.append(acc)
    # 打印top1 ,top2, top3 的准确率追加到res中
    print(res)#[0.5, 1.0, 1.0]
    return None


if __name__ == '__main__':
    tf_sort_argsort()
    tf_topk()
    tf_topk_accuracy()
