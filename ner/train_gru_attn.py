# -*- coding: utf-8 -*-

"""
Train GammaGRUAttn Model
Created on 2020/5/27 
"""

__author__ = "Yihang Wu"

import train
from arguments import GRUAttnArguments as arg

if __name__ == '__main__':
    train.run(arg)
