# -*- coding: utf-8 -*-
"""
__title__="args_test"
__author__="ngc7293"
__mtime__="2020/11/24"
"""
import argparse
import copy
# define hyper-parameters


class A:
    def __init__(self):
        argument = argparse.ArgumentParser()
        argument.add_argument('--embedding', choices=['glove', 'word2vec'], default='glove')
        argument.add_argument('--batch-size-per-gpu', type=int, default=256)
        argument.add_argument('--n-epochs', type=int, default=200)
        argument.add_argument('--lr', type=float, default=1e-5)
        argument.add_argument('--save-dir', type=str, default=None)
        argument.add_argument('--cntn-depth', type=int, default=1)
        argument.add_argument('--cntn-ns', type=int, default=200)
        argument.add_argument('--cntn-k-top', type=int, default=10)
        argument.add_argument('--cntn-r', type=int, default=5)
        argument.add_argument('--dataset', choices=['qnli', 'rte', 'snli', 'mnli'], default='qnli')
        argument.add_argument('--device', type=list, default=[1, 2])

        arg = argument.parse_args()

a = A()

print(a)