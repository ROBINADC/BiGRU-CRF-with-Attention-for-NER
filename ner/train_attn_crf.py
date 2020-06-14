# -*- coding: utf-8 -*-

"""
Train GammaAttnCRF Model
Created on 2020/5/16 
"""

__author__ = "Yihang Wu"

import train
from arguments import AttnCRFArguments as arg

if __name__ == '__main__':
    train.run(arg)
