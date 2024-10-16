# -*- coding: utf-8 -*-
# CreateTime:  2024/2/23 11:50
# Author:      weidafeng@lixlaing.com
# File:        dummy_gpu_task.py
# Software:    PyCharm
# Notes:

import torch
import time
import os
import argparse
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    parser.add_argument('--gpus', help='gpu amount', required=True, type=int)
    parser.add_argument('--size', help='matrix size', type=int, default=25600)
    parser.add_argument('--interval', help='sleep interval', type=float, default=0.01)
    args = parser.parse_args()
    return args


def matrix_multiplication(args):
    a_list, b_list, result = [], [], []
    size = (args.size, args.size)

    for i in range(args.gpus):
        a_list.append(torch.rand(size, device=i))
        b_list.append(torch.rand(size, device=i))
        result.append(torch.rand(size, device=i))

    while True:
        for i in range(args.gpus):
            result[i] = a_list[i] * b_list[i]
        time.sleep(args.interval)


if __name__ == "__main__":
    # usage: python dummy_gpu_task.py --size 25600 --gpus 8 --interval 0.01
    args = parse_args()
    matrix_multiplication(args)