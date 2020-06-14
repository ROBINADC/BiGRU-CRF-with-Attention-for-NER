# -*- coding: utf-8 -*-

"""
Train GammaGRUAttnCRF Model
Created on 2020/5/26 
"""

__author__ = "Yihang Wu"

import train
from arguments import GRUAttnCRFArguments as arg

if __name__ == '__main__':
    train.run(arg)
