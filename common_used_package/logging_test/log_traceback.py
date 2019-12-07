# encoding: utf-8

"""
@author: sunxianpeng
@file: log_traceback3.py
@time: 2019/12/1 19:34
"""
import logging

if __name__ == '__main__':
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("Start print log")
    logger.debug("Do something")
    logger.warning("Something maybe fail.")
    try:
        open("traceback.txt", "rb")
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        logger.error("Faild to open traceback.txt from logger.error", exc_info=True)
    logger.info("Finish")