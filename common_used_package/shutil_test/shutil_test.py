# -*- coding: utf-8 -*-
# @Time : 2019/12/18 23:06
# @Author : sxp
# @Email : 
# @File : shutil_test.py
# @Project : python_common
import shutil

def dir_about():
    dir1 = r'dir1/'
    dir2 = r'dir2/'
    dir3 = r'dir3/'

    """1、递归删除一个目录以及目录内的所有内容，指定ignore_errors，目录不存在时就不会报错"""
    shutil.rmtree(dir2, ignore_errors=True)
    shutil.rmtree(dir3, ignore_errors=True)
    """2、将dir1整个文件夹及以下的内容复制到dir2中，注意：dir2目录不能存在，忽略.pyc文件，制定复制方法"""
    # 如果第3个参数是True，则复制目录时将保持文件夹下的符号连接，如果第3个参数是False，则将在复制的目录下生成物理副本来替代符号连接
    shutil.copytree(dir1,dir2,symlinks=True,ignore=shutil.ignore_patterns("*.pyc",'tmp*'), copy_function=shutil.copy2, ignore_dangling_symlinks=True)
    """3、将dir2文件夹及其下面的文件移动到dir3，注意：dir3不存在时相当于将dir2重命名为dir3，dir3存在时表示将dir2作为dir3的子目录 """
    shutil.move(dir2,dir3)

def file_about():
    file1 = r'file1.txt'
    file2 = r'file2.txt'
    file3 = r'file3.txt'
    shutil.copy2(file1,file2)#将file1的文件内容复制到file2中，会覆盖file2

    """将file2移动到file3中，file3可以不存在，也可以存在 """
    shutil.move(file2,file3)

    """压缩文件"""
    # shutil.make_archive(base_name, format, base_dir)
    # 　　　　　base_name - --> 创建的目标文件名，包括路径，减去任何特定格式的扩展
    # 　　　　　format - --> 压缩包格式后缀：zip、tar、bztar、gztar
    # 　　　　　base_dir - --> 压缩文件的目标文件夹，. 指当前目录，目标文件存在则会覆盖
    # 将当前目录，即shutil目录下所有文件和文件夹压缩在zip_text.zip中
    shutil.make_archive('zip_test', 'zip','.')
    """解压文件"""
    # 将zip_test.zip文件解压缩到zip_unpack/文件夹下
    shutil.unpack_archive('zip_test.zip','zip_unpack/')

def disk_about():
    total, used, free = shutil.disk_usage(".")
    print("当前磁盘共: %iGB, 已使用: %iGB, 剩余: %iGB"%(total / 1073741824, used / 1073741824, free / 1073741824))

def shutil_func():
    print("################################shutil_func###########################")
    # 文件和目录操作
    """1、shutil.copyfileobj(fsrc, fdst[, length]) """
    # 拷贝文件内容, 将fsrc文件里的内容copy到fdst文件中, length:缓冲区大小
    shutil.copyfileobj(open('file.txt', 'r'), open('temp.txt', 'w'))
    """2、拷贝文件内容, 同copyfileobj, 如果dst=src,抛SameFileError异常, dst存在则替换"""
    # """2、shutil.copyfile(src, dst, *, follow_symlinks=True)"""
    dst = shutil.copyfile('file.txt', 'temp.txt')
    """3、仅拷贝权限, 其他信息不受影响"""
    # shutil.copymode(src, dst, *, follow_symlinks=True) //
    shutil.copymode('file.txt', 'temp.txt')
    """4、拷贝状态(权限 / 最后访问时间 / 上次修改时间 / 标志), 其他不受迎影响"""
    # shutil.copystat(src, dst, *, follow_symlinks=True) //
    shutil.copystat('file.txt', 'temp.txt')
    """5、拷贝文件(数据 / 权限)"""
    # shutil.copy(src, dst, *, follow_symlinks=True) //
    dst = shutil.copy('file.txt', 'temp.txt')
    """6、拷贝文件(尝试保留所有元数据) (不能拷贝创建时间,该时间可通过修改系统时间再创建文件来实现)"""
    # shutil.copy2(src, dst, *, follow_symlinks=True) //
    dst = shutil.copy2('file.txt', 'temp.txt')
    """7、忽略的文件, copy_function=自定义复制函数, ignore_dangling_symlinks:True(忽略文件不存在异常) / False(错误列表中添加异常)"""
    # shutil.ignore_patterns(*patterns)
    # symlinks:True(复制链接) / False(复制文件), ignore=ignore_patterns("") //
    """8、递归的复制根目录下的整个目录树"""
    # shutil.copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False) //
    dst = shutil.copytree("root", "temp", symlinks=False, ignore=shutil.ignore_patterns("*.pyc"), copy_function=shutil.copy2, ignore_dangling_symlinks=True)
    """9、删除整个目录树, ignore_errors:是否忽略删除失败错误, onerror=def error(func, path, excinfo)"""
    # shutil.rmtree(path, ignore_errors=False, onerror=None) //
    shutil.rmtree("temp", ignore_errors=True)
    """10、递归移动文件/目录, 目录存在则移动目录, 文件存在则覆盖"""
    # shutil.move(src, dst, copy_function=copy2) //
    dst = shutil.move("root", "temp", copy_function=shutil.copy2)
    """11、给定路径的磁盘使用情况统计信息"""
    total, used, free = shutil.disk_usage(".")  #
    """12、修改用户和组 (Unix可用)"""
    # shutil.chown(path, user=None, group=None) //
    """13、可执行文件路径, path:要查找的路径,未指定使用os.environ的结果"""
    # shutil.which(cmd, mode=os.F_OK | os.X_OK, path=None) //
    path_str = shutil.which("python")


if __name__ == '__main__':
    file_about()
    dir_about()
    disk_about()