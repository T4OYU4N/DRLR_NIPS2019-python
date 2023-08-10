# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:28:09 2023

@author: A
"""
import numpy as np

def ACG(ATb, ATA, lambd, x):
    tol = 1e-5
    g = np.dot(ATA, x) - ATb
    index_lower = (x >= lambd)
    index_upper = (x <= -lambd)
    index_all = index_lower | index_upper
    boundset = index_all & (-x * g >= 0)
    r = -g
    r[boundset] = 0
    gamma = np.dot(r.T, r)

    for k in range(5000):
        boundset_old = boundset
        boundset = index_all & (-x * g >= 0)
        r = -g
        r[boundset] = 0

        if np.all(r == 0):
            break

        gamma1 = gamma
        gamma = np.linalg.norm(r)**2

        if k == 0 or len(boundset_old) != len(boundset) or not np.all(boundset == boundset_old):
            p = r
        else:
            beta = gamma / gamma1
            p = r + beta * p

        q = np.dot(ATA, p)
        alpha = gamma / np.dot(p.T, q)
        x_k_1 = x
        x = x + alpha * p
        index_lower = (x >= lambd)
        index_upper = (x <= -lambd)
        x[index_lower] = lambd
        x[index_upper] = -lambd
        index_all = index_lower | index_upper

        if np.sum(index_all) == 0:
            g = g + alpha * q
        else:
            g = np.dot(ATA, x) - ATb

        if np.linalg.norm(x - x_k_1) < tol:
            break

    return x