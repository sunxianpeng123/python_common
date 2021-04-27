# encoding: utf-8

"""
@author: sunxianpeng
@file: sum_closest.py
@time: 2021/4/27 14:18
"""
import math

"""
给定一个数组，要求在这个数组中找出 3 个数之和离 target 最近。
题目要求找到与目标值 \textit{target}target 最接近的三元组，这里的「最接近」即为差值的绝对值最小。

输入：nums = [-1,2,1,-4], target = 1
输出：2
解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
"""

def threeSumClosest(nums, target):
    """
    :param nums:
    :param target:
    :return:
    """
    nums.sort()
    best = nums[0] + nums[1] + nums[2]#初始值
    for i in range(len(nums)):
        start, end = i + 1, len(nums) - 1
        while start < end:
            sum = nums[start] + nums[end] + nums[i]
            if abs(target - sum) < abs(target - best):
                best = sum
            if sum > target:#比tatget大则向小的方向前进
                end -= 1
            elif sum < target:#向大的方向前进
                start += 1
    return best


if __name__ == '__main__':
    nums = [-1, 2, 1, -4]
    target = 1
    sum = threeSumClosest(nums, target)
    print("sum = {}".format(sum))
