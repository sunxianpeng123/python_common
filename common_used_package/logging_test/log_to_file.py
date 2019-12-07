# encoding: utf-8

"""
@author: sunxianpeng
@file: log_to_file.py
@time: 2019/12/1 19:16
"""
import logging
def log_to_file_not_control():
    """# 将日志写入当前路径下的 log.txt，没有文件会新建,不会将日志打印在控制台"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")

def log_to_file_and_control():
    """ogger中添加StreamHandler，可以将日志输出到屏幕上，"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    logger.info("Finish")

if __name__ == '__main__':
    # 将日志写入当前路径下的 log.txt，没有文件会新建,不会将日志打印在控制台
    # log_to_file_not_control()
    log_to_file_and_control()
