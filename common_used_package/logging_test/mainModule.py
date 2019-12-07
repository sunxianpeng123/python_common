# encoding: utf-8

"""
@author: sunxianpeng
@file: mainModule.py
@time: 2019/12/1 19:37
"""
import logging
import common_used_package
from common_used_package.logging_test import subModule


if __name__ == '__main__':
    logger = logging.getLogger("mainModule")
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

    logger.info("creating an instance of subModule.subModuleClass")
    a = subModule.SubModuleClass()
    logger.info("calling subModule.subModuleClass.doSomething")
    a.doSomething()
    logger.info("done with  subModule.subModuleClass.doSomething")
    logger.info("calling subModule.some_function")
    subModule.som_function()
    logger.info("done with subModule.some_function")