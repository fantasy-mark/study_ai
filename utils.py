#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/31 11:20
# @Author  : F1243749 Mark
# @File    : utils.py
# @Depart  : NPI-SW
# @Desc    :
from time import time

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def show_sample(train_dataset):
    num_of_images = 40
    for index in range(1, num_of_images + 1):
        plt.subplot(4, 10, index)
        plt.axis('off')
        plt.imshow(train_dataset.data[index], cmap='gray_r')
    plt.show()


def mnist_dataset(transform, batch_size=64):
    # 如果设置 download=True 下载失败可以按照 data/tips.txt 网址下载放入到 data/MNIST/raw/
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   download=False,
                                   transform=transform)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size)
    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  download=False,
                                  transform=transform)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=batch_size)

    return train_dataset, test_dataset, train_loader, test_loader
