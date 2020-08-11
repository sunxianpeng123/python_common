# # -*- coding: utf-8 -*-
# # @Time : 2020/7/23 19:35
# # @Author : sxp
# # @Email :
# # @File : split_group.py
# # @Project : python_common
#
#
# import os
# import shutil
#
#
# def mymovefile(srcfile, dstfile):
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % (srcfile))
#     else:
#         fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
#         if not os.path.exists(fpath):
#             os.makedirs(fpath)  # 创建路径
#         shutil.move(srcfile, dstfile)  # 移动文件
#         print("move %s -> %s" % (srcfile, dstfile))
#
#
# def mycopyfile(srcfile, dstfile):
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % (srcfile))
#     else:
#         fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
#         if not os.path.exists(fpath):
#             os.makedirs(fpath)  # 创建路径
#         shutil.copyfile(srcfile, dstfile)  # 复制文件
#         print("copy %s -> %s" % (srcfile, dstfile))
#
#
# def group_rules(source_dir, target_dir, group_num):
#     """"""
#     if not os.path.exists(target_dir): os.mkdir(target_dir)
#     if not os.path.isdir(source_dir): raise Exception("source_dir is not dir")
#     if not os.path.isdir(target_dir): raise Exception("target_dir is not dir")
#
#     files = os.listdir(source_dir)
#     all_file_num = len(files)
#     n = int(all_file_num / group_num)
#     remain = all_file_num - group_num * n
#     print("all_file_num = {},n = {},remain={}".format(all_file_num, n, remain))
#     # 移动整数组
#     for i in range(n):
#         area = (i+1)*15
#         for j in range(1,area+1):
#             filename = "task_" + str(j)+ ".csv"
#             filepath = os.path.join(source_dir,filename)
#             target_inside_dir = os.path.join(target_dir,"group_"+str(i+1))
#
#             if not os.path.exists(target_inside_dir):os.mkdir(target_inside_dir)
#             filepath_target = os.path.join(target_inside_dir,filename)
#             mycopyfile(filepath,filepath_target)
#     # 收尾
#     if remain != 0:
#         area = n * 15
#         for j in range(area, area+remain+1):
#             filename = "task_" + str(j) + ".csv"
#             filepath = os.path.join(source_dir, filename)
#             target_inside_dir = os.path.join(target_dir, "group_" + str( + 1))
#
#             if not os.path.exists(target_inside_dir): os.mkdir(target_inside_dir)
#             filepath_target = os.path.join(target_inside_dir, filename)
#             mycopyfile(filepath, filepath_target)
#
#
#
#
# if __name__ == '__main__':
#     source_dir = r"C:\Users\sxp\Desktop\最新关注任务"
#     target_dir = r"C:\Users\sxp\Desktop\最新关注任务_grouped"
#     group_num = 15
#     group_rules(source_dir, target_dir, group_num)
