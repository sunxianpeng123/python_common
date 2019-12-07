# encoding: utf-8

"""
@author: sunxianpeng
@file: fizz_buzz.py
@time: 2019/11/7 13:38
"""
import numpy as np
import torch
"""FizzBuzz
FizzBuzz是一个简单的小游戏。游戏规则如下：
    从1开始往上数数，当遇到3的倍数的时候，说fizz，
    当遇到5的倍数，说buzz，
    当遇到15的倍数，就说fizzbuzz，其他情况下则正常数数。
我们可以写一个简单的小程序来决定要返回正常数值还是fizz, buzz 或者 fizzbuzz。
"""
#
def fizz_buzz_encode(i):
    if i % 15 ==0: return 3
    elif i % 5 ==0: return 2
    elif i % 3 ==0: return 1
    else:           return 0

def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]


def binary_encode(i,num_digits):
    # 将数字进行二进制反序列表示,切维度为 num_digits
    return np.array([i >> d & 1 for d in range(num_digits)])

if __name__ == '__main__':
    """1、基于规则的实现"""
    print(fizz_buzz_decode(1, fizz_buzz_encode(1)))
    """2、基于神经网络实现"""
    num_digits = 10
    binary = binary_encode(1,num_digits)
    print("binary = {}".format(binary))

    # 训练数据
    trX = torch.Tensor([binary_encode(i,num_digits) for i in range(101,2 ** num_digits)])
    trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2 ** num_digits)])
    print(trX.shape,trY.shape)
    # 定义模型
    num_hidden = 100
    model = torch.nn.Sequential(
        torch.nn.Linear(num_digits,num_hidden),#第一层输入 (num_digits,num_hidden)维
        torch.nn.ReLU(),
        torch.nn.Linear(num_hidden, 4)#第二层 (num_hidden, 4)，最后输出四维
    )
    # 由于FizzBuzz游戏本质上是一个分类问题，我们选用Cross Entropyy Loss函数。
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    # 训练模型
    batch_size = 120
    for epoch in range(1000):
        for start in range(0,len(trX),batch_size):
            # 取当前批次数据
            end = start + batch_size
            batchX = trX[start:end]
            batchY = trY[start:end]
            # 训练当前批次
            if torch.cuda.is_available():
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            y_pred = model(batchX)
            loss = loss_fn(y_pred,batchY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss_fn(model(trX),trY).item()
        print("Epoch = {}, Loss = {}".format(epoch,loss))

    # 预测
    testX = torch.Tensor([binary_encode(i,num_digits) for i in range(1,101)])
    if torch.cuda.is_available():
        testX =testX.cuda()
    with torch.no_grad():#预测过程不保存梯度
        testY = model(testX)
        zip()
    # 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))
    print(type(testY))
    # 按维度dim返回最大值，并且返回索引。 torch.max(a, 0),返回每一列中最大值的那个元素，
    # 且返回索引（返回最大元素在这一列的行索引）。返回的最大值和索引各是一个tensor，
    # 一起构成元组(Tensor, LongTensor)
    # print(testY.max(dim=1))
    # print(testY.max(1)[1])
    # print(list(testY.max(1)[1].data.tolist()))
    print([fizz_buzz_decode(i, x) for (i, x) in predictions])
