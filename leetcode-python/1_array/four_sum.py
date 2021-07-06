# encoding: utf-8

"""
@author: sunxianpeng
@file: four_sum.py
@time: 2021/4/27 18:34
"""

"""
给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

输入：nums = [1, 0, -1, 0, -2, 2], 
      target = 0
输出：[[-2, -1, 1, 2],
       [-2, 0, 0, 2],
       [-1, 0, 0, 1]]

输入：nums = [], target = 0
输出：[]
"""


def fourSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[List[int]]
    """
    result = []
    if len(nums) < 4:
        return result
    nums.sort()
    for i in range(len(nums) - 3):
        # 当前值和前一个值相等，则跳过当前值，继续向下执行
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        sum = nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3]
        if sum > target:
            break
        if sum == 1:
            pass

    pass


if __name__ == '__main__':
    nums = [1, 0, -1, 0, -2, 2]
    target = 0