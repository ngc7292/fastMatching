# -*- coding: utf-8 -*-
"""
__title__="matching-dcbert"
__author__="ngc7293"
__mtime__="2021/1/14"
"""
import argparse

class DCBERTConfig:
    def __init__(self):
        arg = argparse.ArgumentParser()
        arg.add_argument("--dataset")