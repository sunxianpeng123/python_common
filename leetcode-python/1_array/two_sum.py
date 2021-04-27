# encoding: utf-8

"""
@author: sunxianpeng
@file: two_sum.py
@time: 2021/4/26 15:49
"""

"""
在数组中找到 2 个数之和等于给定值的数字，结果返回 2 个数字在数组中的下标。

输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

输入：nums = [3,2,4], target = 6
输出：[1,2]

"""
def twoSum(nums,targets):
    """ """
    hashmap = {}
    res = []
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for index,num in enumerate(nums):
        # print("index = {},, num = {}".format(index,num))
        hashmap[num] = index
    for index, num in enumerate(nums):
        last = target - num
        index_last = hashmap.get(last)
        if last in hashmap and index != index_last:
            res = [index,index_last]
            break
    return res



if __name__ == '__main__':
    nums = [2,7,11,15]
    target = 9
    indexs = twoSum(nums,target)
    print(indexs)