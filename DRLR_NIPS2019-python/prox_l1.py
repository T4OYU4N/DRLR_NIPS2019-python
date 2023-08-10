# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:35:27 2023
 
@author: A
"""
import numpy as np

def prox_l1(v, lambd):
    x = np.maximum(0, v - lambd) - np.maximum(0, -v - lambd)
    return x