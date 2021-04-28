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
        # 第一个数字大于0时，三数之和不可能=0了，直接返回
        if nums[i] > 0:
            return result
        # 需要和上一次枚举的数不相同
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        start, end = i + 1, len(nums) - 1
        while start < end:
            sum = nums[start] + nums[end] + nums[i]
            if sum == 0:
                result.append([nums[i], nums[start], nums[end]])
                # 移动start索引，直到找到和当前start对应的值不想等的下一个start值
                while start < end and nums[start] == nums[start + 1]:
                    start += 1
                # 移动end索引，直到找到和当前end对应的值不想等的下一个end值
                while start < end and nums[end] == nums[end - 1]:
                    end -= 1
                start += 1
                end -= 1
            #当前和大于零则移动右侧索引，使之变小
            elif sum > 0:
                end -= 1
            # 当前值小于零则移动左侧索引，使之变大
            else:
                start += 1
    return result


if __name__ == '__main__':
    nums = [-1, 0, 1, 2, -1, -4]
    result = threeSum(nums)
    for r in result:
        print(r)
