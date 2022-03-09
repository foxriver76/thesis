#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:46:11 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import numpy as np

def elm_kernel(X, Y):
    """ELM KERNEL -  between two matrices
    Frenay, Parameter-insensitive kernel in extreme learning for non-linear supportvector regression
    Eq. 15
    """
    # here sigma^2 = 1e10 should be large enough to dont add errors
    dSig = 1 / (2 * 1e10)
    # sums per row
    KxDiag = np.sum(X * X, axis=1).reshape((-1, 1))
    KyDiag = np.sum(Y * Y, axis=1).reshape((-1, 1))
    
    return 2 / np.pi * np.arcsin((1 + X @ Y.T) /
                              np.sqrt((dSig + 1 + KxDiag) @ (dSig + 1 + KyDiag).T))
    
def elm_kernel_vec(x, y):
    """ELM KERNEL - between two vectors
    Frenay, Parameter-insensitive kernel in extreme learning for non-linear supportvector regression
    Eq. 15
    """
    # here sigma^2 = 1e10 should be large enough to dont add errors
    dSig = 1 / (2 * 1e10)
    # sums per row
    
    return 2 / np.pi * np.arcsin((1 + x @ y) /
                              np.sqrt((dSig + 1 + x @ x) * (dSig + 1 + y @ y).T))



"""Test"""
if __name__ == '__main__':
    X = np.array([
            [17, 24, 1, 8, 15],
            [23, 5, 7, 14, 16],
            [4, 6, 13, 20, 22],
            [10, 12, 19, 21, 3],
            [11, 18, 25, 2, 9]
         ])
    
    print(elm_kernel(X, X))
    
    print(elm_kernel_vec(np.array([1, 2, 3]), np.array([100, 0.5, 0.3])))