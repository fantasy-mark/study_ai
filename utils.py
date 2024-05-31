#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/31 11:20
# @Author  : F1243749 Mark
# @File    : utils.py
# @Depart  : NPI-SW
# @Desc    :
from time import time


def time_cost(func):
    """
    装饰器，用于统计函数执行耗时，用法：在函数上加上 @time_cost
    @param func: 函数指针
    @return:
    """
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        if elapsed_time > 60.:
            minuted = elapsed_time // 60
            second = elapsed_time % 60
            print(f"Function took {int(minuted)}m {int(second)}s to execute.")
        else:
            print(f"Function {func.__name__} took {elapsed_time:.2f}s to execute.")

        return result

    return wrapper
