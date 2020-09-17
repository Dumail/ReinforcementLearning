#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/9/15 9:23
# @Author  : PCF
# @Email   : pan.chaofan@foxmail.com
# @File    : log.py
# @Software: PyCharm
import logging
from functools import wraps


# TODO USE
def called_log(func):
    @wraps(func)
    def with_logging(*arg, **kwargs):
        logging.debug(func.__name__ + ' was called')
        return func(*arg, **kwargs)

    return with_logging()
