import logging
import time
import os
import sys

if not os.path.exists("./result"):
    os.mkdir("./result")
if not os.path.exists("./result/log"):
    os.mkdir("./result/log")

def init_logger(logger=None):
    if logger == None:
        logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",datefmt="%Y-%m-%d:%H:%M:%S")
    file_handler = logging.FileHandler(filename="./result/log/" + time.strftime("%Y_%m_%d_%H_%M", time.localtime()) +".log", mode="a")
    file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger

log = init_logger()

