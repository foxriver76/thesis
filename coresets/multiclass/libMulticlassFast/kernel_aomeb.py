#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:42:18 2020

@author: moritz
"""

from typing import Type
from .kernel_core_meb import KernelCoreMEB
import numpy as np
from .elm_kernel import elm_kernel_vec

class Kernel_AOMEB:
    
#    def __init__(self, init_point_set:[]=None, eps:float=None, append_mode:bool=False, idx:int=None, inst:Type['AOMEB']=None, gamma:float=0.1):
    def __init__(self, init_point_set:[]=None, eps:float=None, idx:int=None, inst:Type['AOMEB']=None, kernel_fun=elm_kernel_vec):
#        self.gamma = gamma
        self.idx = init_point_set[0]['idx']
        self.eps = eps

        self.radius2 = 0.0
        self.c_norm = 0.0
        
        self.kernel_cache = []
        self.kernel_fun = kernel_fun

        self.coreset_initial(init_point_set)
            
    def coreset_initial(self, points:[]) -> None:
        init_coreset = KernelCoreMEB(points, self.eps)

        self.radius2 = init_coreset.radius2
        self.c_norm = init_coreset.c_norm
        
        self.core_points = init_coreset.points
        self.coefficents = np.array(list(init_coreset.coefficents.values()))

        self.init_kernel_matrix()
        
    def init_kernel_matrix(self):
        for i in range(len(self.core_points)):
            kernel_vector = [self.kernel_eval(self.core_points[i], self.core_points[j]) for j in range(len(self.core_points))]
  
            self.kernel_cache.insert(i, kernel_vector)
            
    def append(self, points:[]) -> None:
        new_core_points = []
        
        for point in points:
            # if point outside, we need to add it to the coreset
            if self._dist2_wc(point) > (1.0 + self.eps) * (1.0 + self.eps) * self.radius2:
                new_core_points.append(point)
                
        if len(new_core_points) > 0:
            self.solve_apx_ball(new_core_points)
                        
    def solve_apx_ball(self, points:[]) -> None:
        for point in points:
            self.core_points.append(point)
#            self.coefficents.append(0.0)
            self.coefficents = np.concatenate((self.coefficents, [0.0]))
            self.add_to_kernel_matrix(point)
            
        furthest_pair = self.find_farthest_point()
        nearest_pair = self.find_nearest_point()
        
        # calc max delta from our ''center''
#        delta_plus = furthest_pair['dist2'] / (self.radius2 + 1e-20)
#        delta_minus = nearest_pair['dist2'] / (self.radius2 + 1e-20)
        # rel Abweichung zum Radius also z. b. 10 % ist 1.1 daher ziehe 1 ab
        delta_plus = furthest_pair['dist2'] / (self.radius2 + 1e-20) - 1.0
        delta_minus = 1.0 - nearest_pair['dist2'] / (self.radius2 + 1e-20)
        delta = max(delta_plus, delta_minus)
        
        while delta > (1.0 + self.eps) * (1.0 + self.eps) - 1.0:
#        if delta > (1.0 + self.eps) * (1.0 + self.eps) - 1.0:
            if delta > delta_minus:
                # lambda relative abweichung, um den punkt der draußen liegt höher zu gewichten
                _lambda = delta / (2.0 * (1.0 + delta))
                
                # 1 here fixed, coeffs should sum up to 1
                self.coefficents = (1.0 - _lambda) * self.coefficents
                        
                self.coefficents[furthest_pair['idx']] = self.coefficents[furthest_pair['idx']] + _lambda
            else:
                lambda1 = delta_minus / (2.0  * (1.0 - delta_minus))
                lambda2 = self.coefficents[nearest_pair['idx']] / (1.0 - self.coefficents[nearest_pair['idx']] + 1e-20)
                _lambda = min(lambda1, lambda2)
                
#                print(_lambda)
                # if this gets zero coeffs stay the same
                if _lambda == 0: raise ValueError(max(lambda1, lambda2))
                # 1 here fixed, coeffs should sum up to 1
                self.coefficents = (1.0 + _lambda) * self.coefficents

                self.coefficents[nearest_pair['idx']] = self.coefficents[nearest_pair['idx']] - _lambda
                
            # update c norm
            self.c_norm = np.sum(np.outer(self.coefficents, self.coefficents) * np.array(self.kernel_cache))
            
            self.radius2 = 3.0 - self.c_norm
            
            furthest_pair = self.find_farthest_point()
            nearest_pair = self.find_nearest_point()
            
#            delta_plus = furthest_pair['dist2'] / (self.radius2 + 1e-20)
#            delta_minus = nearest_pair['dist2'] / (self.radius2 + 1e-20)
            delta_plus = furthest_pair['dist2'] / (self.radius2 + 1e-20) - 1.0
            delta_minus = 1.0 - nearest_pair['dist2'] / (self.radius2 + 1e-20)
            delta = max(delta_plus, delta_minus)
                
    def add_to_kernel_matrix(self, point:[]) -> None:
        kernel_vector = []
        for i in range(len(self.core_points) - 1):
            value = self.kernel_eval(point, self.core_points[i])
            kernel_vector.append(value)
            self.kernel_cache[i].append(value)
            
        kernel_vector.append(self.kernel_eval(point, point))
        self.kernel_cache.insert(len(self.core_points) - 1, kernel_vector)
    
    def find_nearest_point(self) -> {}:
        dist = self._dist2_wc_idx_m()
        idx = np.argmin(dist)
            
        return {"idx": idx, "dist2": dist[idx]}
    
    def find_farthest_point(self) -> {}:
        dist = self._dist2_wc_idx_m()
        idx = np.argmax(dist)
            
        return {"idx": idx, "dist2": dist[idx]}
         
    def kernel_eval(self, p1:[], p2:[]) -> float:
        return KernelCoreMEB.kernel_eval(self, p1, p2)

    def _dist2_wc(self, point:[]) -> float:
        dist2 = np.sum([self.coefficents[i] * self.kernel_eval(self.core_points[i], point) for i in range(len(self.core_points))])

        return 3.0 - 2.0 * dist2 + self.c_norm # Eq12
#        return 1.0 + self.c_norm - 2.0 * dist2
        
    def _dist2_wc_idx_m(self) -> float:
        dist2_v = np.sum(self.coefficents * np.array(self.kernel_cache), axis=1)
        return 3.0 - 2.0 * dist2_v + self.c_norm
