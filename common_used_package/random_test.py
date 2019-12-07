# encoding: utf-8

"""
@author: sunxianpeng
@file: random.py
@time: 2019/12/2 17:39
"""
import random
def random_float_num():
    """随机生成浮点数:有两种，一种没有参数,默认（0-1），另一种可以指定随机生成的浮点数范围。"""
    print("######################随机生成浮点数#######################")
    #
    # #从0~1中间随机产生一个小数点后16位的浮点数
    print('random.random = {}'.format(random.random()))
    # 从1~3中间随机产生一个小数点后16位的浮点数
    print('random.uniform = {}'.format(random.uniform(10,20)))
    # random.random = 0.21846360217280592
    # random.uniform = 11.69708818020138

def random_range_int():
    """随机生成指定范围的整数：有两种方法，第二种除了可以指定范围，还可以指定步长。"""
    print("###################随机生成指定范围的整数#####################")
    print('random.randint = {}'.format(random.randint(1,10)))
    print('random.randrange = {}'.format(random.randrange(10, 20)))
    print('random.randrange = {}'.format(random.randrange(10, 20,5)))
    #random.randint = 4
    # random.randrange = 19
    # random.randrange = 15

def random_choice():
    """随机生成指定样式中的元素：样式可以是字符串、元组、列表。"""
    print("###################随机生成指定样式中的元素######################")
    # 从列表中随机选择一个元素
    list_1 = [1,2,'a','b']
    print('random.choice list_1 = {}'.format(random.choice(list_1)))
    # 从列表中随机选择一个元素组成一个新列表
    list_2 = [1,2,3,4]
    print('random.choices list_2 = {}'.format(random.choices(list_2)))
    str_1 = 'abcdefg'
    print('random.choice str_1  = {}'.format(random.choice(str_1)))
    tup_1 = (1,2,3,4)
    print('random.choice tup_1  = {}'.format(random.choice(tup_1)))
    #random.choice list_1 = 2
    # random.choice list_2 = 1
    # random.choice str_1  = d
    # random.choice tup_1  = 4

def random_sample():
    """随机生成指定数目的指定样式中的元素：样式可以是字符串、元组、列表、集合。"""
    print("#################随机生成指定数目的指定样式中的元素###################")
    # 必须指定选择的元素个数,
    str_1 = 'abcdefg'
    print('random.sample str_1 = {}'.format(random.sample(str_1,2)))
    tup_1 = (1,2,3,4)
    print('random.sample tup_1 = {}'.format(random.sample(tup_1, 3)))
    list_1 = [1,2,3,4,5,6,7,8,9]
    print('random.sample list_1 = {}'.format(random.sample(list_1, 3)))
    #random.sample str_1 = ['a', 'd']
    # random.sample tup_1 = [3, 2, 4]
    # random.sample list_1 = [4, 5, 2]

def random_shuffle():
    """将列表的元素的顺序打乱：类似于生活中的洗牌，此方法返回值为空，将改变原来列表"""
    print("######################将列表的元素的顺序打乱######################")
    item = [1,2,3,4,5,6]
    print('before shuffle sequence = {}'.format(item))
    random.shuffle(item)
    print('after shuffle sequence = {}'.format(item))
    #before shuffle sequence = [1, 2, 3, 4, 5, 6]
    # after shuffle sequence = [6, 3, 4, 2, 1, 5]

def getSixNumVerificationCode():
    print("###################随机生成六位数字验证码####################")
    """随机生成六位数字验证码"""
    captcha = ''
    for i in range(6):
        num = random.randint(0,9)
        captcha += str(num)
    print(captcha)# 886515

if __name__ == '__main__':
    #1、 随机生成浮点数
    random_float_num()
    #2、随机生成指定范围的整数
    random_range_int()
    # 3、随机生成指定样式中的元素
    random_choice()
    # 4、随机生成指定数目的指定样式中的元素
    random_sample()
    # 5、将列表的元素的顺序打乱
    random_shuffle()
    # 6、随机生成六位数字验证码
    getSixNumVerificationCode()



