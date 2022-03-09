#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 07:58:47 2020

@author: moritz
"""

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
from .swmeb_plus import SWMEB_Plus

class SWMEBWIN(BaseDriftDetector):
    """Concept Drift Detector based on MEB
        epsilon : float, epsilon for eps-MEB
        threshold : int, threshold for stat_size - how many dps have to be outside of ball
        kernel_func : function, kernel function used to calculate MEB
        w_size : int, window size for MEB
        state_size : int, window size for test chunk
    """
    
    def __init__(self, classes:[], epsilon=0.1, w_size=100):
        self.eps = epsilon
        self.change_detected = False
        self.w_size = w_size
        self.detectors = {}
        for cl in classes:
            self.detectors[cl] = {"detector": SWMEB_Plus(eps1=epsilon, window_size=w_size, batch_size=1), "n": 0, "_n": 0}
        
    def add_element(self, value:np.array, label: any) -> None:
        """Adds element to window
        value : array-like, datapoint to add to window
        """
         
        data = [{"idx": self.detectors[label]['_n'], "data": value}]
        self.detectors[label]['n'] += 1
        self.detectors[label]['_n'] += 1
        # now lets take our instance and check for drift
        if self.detectors[label]['n'] > self.w_size:
            inst = self.detectors[label]['detector'].instances[self.detectors[label]['detector'].index[0]]
            dist = self._dist(value, inst.center)
            if dist < (inst.radius * (1.0 + self.eps)):
                self.change_detected = False
            else:
                self.change_detected = True
                # ball is not appropriate anymore, reset n of all balls
                for _label in self.detectors:
                    self.detectors[_label]['n'] = 0
        else:
            self.change_detected = False
        
        # now append our data
        self.detectors[label]['detector'].append(data)
        
    @property           
    def detected_change(self):
        return self.change_detected
    
    def _dist(self, x:[], y:[]) -> float:
        """Calculate euclidean distance between two points"""
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))