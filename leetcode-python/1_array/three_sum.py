# encoding: utf-8

"""
@author: sunxianpeng
@file: three_sum.py
@time: 2021/4/27 18:54
"""

"""
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。
给定一个数组，要求在这个数组中找出 3 个数之和为 0 的所有组合。

输入：nums = [-1, 0, 1, 2, -1, -4]
输出：[[-1, -1, 2],
       [-1, 0, 1]]

输入：nums = []
输出：[]
"""


def threeSum(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    result = []
    nums.sort()
    print(nums)
    for i in range(len(nums)):
        start, end = i + 1, len(nums) - 1
        while start < end:
            sum = nums[start] + nums[end] + nums[i]
            if sum == 0:
                result = result + [nums[i], nums[start], nums[end]]
            elif sum > 0:
                end -= 1
            else:
                start += 1

    return result


if __name__ == '__main__':
    nums = [-1, 0, 1, 2, -1, -4]
    result = threeSum(nums)
    for r in result:
        print(r)
