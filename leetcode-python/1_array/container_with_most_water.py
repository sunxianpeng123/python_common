# encoding: utf-8

"""
@author: sunxianpeng
@file: container_with_most_water.py
@time: 2021/4/27 11:17
"""

"""
给出一个非负整数数组 a1，a2，a3，…… an，每个整数标识一个竖立在坐标轴 x 位置的一堵高度为 ai 的墙，
选择两堵墙，和 x 轴构成的容器可以容纳最多的水。

输入：[1,8,6,2,5,4,8,3,7]
输出：49 
解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。


这一题也是对撞指针的思路。首尾分别 2 个指针，每次移动以后都分别判断长宽的乘积是否最大。
"""


def maxArea(height):
    """
    :type height: List[int]
    :rtype: int
    """
    max_area, i, j = 0, 0, len(height) - 1
    while i < j:
        l = j - i
        if height[i] < height[j]:
            h = height[i]
            max_area = max(max_area, h * l)
            i += 1
        else:
            h = height[j]
            max_area = max(max_area, h * l)
            j -= 1
    return max_area

if __name__ == '__main__':
    height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    res = maxArea(height)
    print(res)
