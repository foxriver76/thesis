#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:27:15 2019

@author: moritz
"""

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from meb_matlab import meb_vec, meb_distance
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

class MEBWIN(BaseDriftDetector):
    """Concept Drift Detector based on MEB
        epsilon : float, epsilon for eps-MEB
        threshold : int, threshold for stat_size - how many dps have to be outside of ball
        kernel_func : function, kernel function used to calculate MEB
        w_size : int, window size for MEB
        state_size : int, window size for test chunk
    """
    
    def __init__(self, epsilon=0.1, sensitivity=0.95, kernel_func=linear_kernel, w_size=100, stat_size=30):
        self.eps = epsilon
        self.kernel_func = kernel_func
        self.threshold = int(w_size * round(1 - sensitivity, 8))
        self.window = []
        self.change_detected = False
        self.w_size = w_size
        self.stat_size = stat_size
        
    def add_element(self, value):
        """Adds element to window
        value : array-like, datapoint to add to window
        """
        current_length = len(self.window)
        self.window.insert(current_length, value)
        
        # if window is full, remove oldest element
        if current_length >= self.w_size:
            self.window.pop(0)
            # get win - stat_size oldest dps
            rnd_window = np.array(self.window[:-self.stat_size])
            # get stat_size newest dps
            stat_window = np.array(self.window[-self.stat_size:])
            
            # calc ball of rnd_window and see how many of stat_window are lying outside
            S, R, alphas = meb_vec(rnd_window, self.eps, self.kernel_func)
            
            vAlpha = np.array(alphas).ravel()
            protos = np.array(rnd_window[S])
            
            mKSS = self.kernel_func(protos, protos)
            
            vDiags = np.diag(self.kernel_func(stat_window, stat_window))
            
            # Sim between protos and new points
            mK = self.kernel_func(protos, stat_window)

            dist = meb_distance(vDiags, vAlpha, mK, mKSS)
            
            # count dps outside of eps-MEB
            count_out_of_meb = np.count_nonzero(dist - R * (1 + self.eps) > 0)
            
            self.change_detected = True if count_out_of_meb > self.threshold else False
            # if new change detected - clear windows
            if self.change_detected:
                self.window = []
        else:
            self.change_detected = False
            
    @property           
    def detected_change(self):
        return self.change_detected
    
    def reset(self):
        self.window = []
        self.change_detected = False