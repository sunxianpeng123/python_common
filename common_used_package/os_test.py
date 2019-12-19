# encoding: utf-8

"""
@author: sunxianpeng
@file: 1_indoor.py
@time: 2019/12/1 14:16
"""
import os
from random import randint
def dir_about(dir_path,file_name,test_mkdir,test_mkdirs):
    """
    操作文件夹的相关方法
    :param dir_path: 用户查看文件夹属性的文件夹路径
    :param file_name: 用于示例join的文件名字
    :param test_mkdir: 用于示例创建单个文件夹的路径
    :param test_mkdirs: 用于示例递归创建文件夹的路径
    :return:
    """
    print("""############################文件夹相关的操作##############################""")
    # 1、功能:查看
    print('1、当前所在路径 = {}'.format(os.getcwd())) # F:\PythonProjects\python_study\common_used_package
    # 2、列举目录下所有的文件,返回的是列表类型
    print('2、目录下所有的文件 = {}'.format(os.listdir(dir_path))) # ['test.txt', 'test1.jpg']
    # 3、功能: 返回path的绝对路径
    print('3、path的绝对路径 = {}'.format((os.path.abspath(dir_path))))#F:\PythonProjects\python_study\common_used_package\data
    # 4、将路径分解为(文件夹, 文件名), 返回的是元组类型。
    print('4、路径分解为(文件夹, 文件名) ={}'.format(os.path.split(dir_path)))# ('./data', '')
    # 5、将path进行组合
    print('5、path进行组合 = {}'.format(os.path.join(dir_path,file_name)))#./data/test.txt
    # 6、返回path中文件夹部分, 不包括”\”
    print('6、path中文件夹部分 = {}'.format(os.path.dirname(dir_path)))#./data
    # 7、功能：判断该路径是否为目录
    print('7、路径是否为目录 = {}'.format(os.path.isdir(dir_path)))#True
    # 创建目录，目录已存在将不能再创建
    print('8、创建目录 = {}'.format(os.mkdir(test_mkdir)))
    # 递归式的创建目录，上级目录不存在，将会创建；目录已存在将不能再创建
    print('9、递归式的创建目录 = {}'.format(os.makedirs(test_mkdirs)))
    # 删除一个空目录
    print('10、删除一个空目录 = {}'.format(os.rmdir(test_mkdir)))
    # 递归删除空目录，例如os.removedirs('dir1/dir2/dir3'), dir1下除了dir2还有其他，dir2下只有dir3，则删除dir3和dir2
    print('11、递归删除空目录 = {}'.format(os.removedirs(test_mkdirs)))
    # 给目录改名
    print('12、给目录改名 = {}'.format(os.renames(dir_path,dir_path)))

def file_about(file_path):
    """
    操作文件的相关方法
    :param file_path: 文件的路径
    :return:
    """
    print("""############################文件相关的操作##############################""")
    # 将路径分解为(文件夹, 文件名), 返回的是元组类型。
    print('1、路径分解为(文件夹, 文件名) = {}'.format(os.path.split(file_path)))# ('./data', 'test.txt')
    # 返回path中文件夹部分, 不包括”\”
    print('2、path中文件夹部分 = {}'.format(os.path.dirname(file_path)))# ./data
    # 功能: 返回path中的文件名
    print('3、path中的文件名 = {}'.format(os.path.basename(file_path)))# test.txt
    # 功能: 获取文件的大小, 若是文件夹则返回0
    print('4、获取文件的大小 = {}'.format(os.path.getsize(file_path)))# 26
    # 功能：判断文件是否存在，若存在返回True，否则返回False
    print('5、文件是否存在 = {}'.format(os.path.exists(file_path))) # True
    # 功能：判断该路径是否为文件
    print('6、路径是否为文件 = {}'.format(os.path.isfile(file_path))) #True
    # 删除指定的文件
    # print('7、删除指定的文件 = {}'.format(os.remove(file_path)))
    # 给文件改名
    print('8、给文件改名 = {}'.format(os.renames(file_path,file_path)))

def os_walk_function(data_dir):
    """
    os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])
    top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
        root 所指的是当前正在遍历的这个文件夹的本身的地址
        dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。
            如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
    onerror -- 可选，需要一个 callable 对象，当 walk 需要异常时，会调用。
    followlinks -- 可选，如果为 True，则会遍历目录下的快捷方式(linux 下是软连接 symbolic link )
                实际所指的目录(默认关闭)，如果为 False，则优先遍历 top 的子目录
    :param data_dir: 用于os.walk()函数遍历的文件夹目录
    :return:
    """
    print("""############################测试os模块的walk函数##############################""")
    i = 0
    for root,dirs,files in os.walk(data_dir,topdown=False):
        print("************第 {} 个root 目录 ************".format(i))
        print('root = {}'.format(root))
        print("显示 root = {} 下的所有文件名字")
        for name in files:
            print(os.path.join(root, name))
        print("显示 root 下的所有文件夹名字")
        for name in dirs:
            print(os.path.join(root, name))
        i += 1


if __name__ == '__main__':
    rand_num_1 = randint(1,10000)
    rand_num_2 = randint(10000, 20000)
    dir_path = './data/'
    file_path = r'./data/test.txt'
    file_name = 'test.txt'
    test_mkdir = './data/test_mkdir/' + str(rand_num_1)
    test_mkdirs = './data/test_mkdirs/' + str(rand_num_2) + '/' + str(1)

    dir_about(dir_path,file_name,test_mkdir,test_mkdirs)
    file_about(file_path)
    os_walk_function('.')

