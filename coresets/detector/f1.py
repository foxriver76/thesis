#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:29:11 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

# ADWIN 5 true positives, 31 false positives 
# ADWIN 3995977 true negatives, 3987 false negatives

# DDM 25 true positives, 531 false positives
# DDM 998471 true negatives, 973 false negatives

# K-MEBWIND 924 true positives, 15837 false positives
# K-MEBWIND 3980171 true negatives, 3068 false negatives

# MEBWIND 885 true positives, 15902 false positives
# MEBWIND 3980106 true negatives, 3107 false negatives

def calc_f1(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall / (precision + recall))

f1_adwin = calc_f1(5, 31, 3995977, 3987)
f1_ddm = calc_f1(25, 531, 998471, 973)
f1_k_mebwind = calc_f1(924, 15837, 3980171, 3068)
f1_mebwind = calc_f1(885, 15902, 3980106, 3107)