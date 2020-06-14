# -*- coding: utf-8 -*-

"""
Train GammaGRUCRF Model
Created on 2020/5/26 
"""

__author__ = "Yihang Wu"

import train
from arguments import GRUCRFArguments as arg

if __name__ == '__main__':
    train.run(arg)
