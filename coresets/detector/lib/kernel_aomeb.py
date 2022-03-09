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
    def __init__(self, init_point_set:[]=None, eps:float=None, append_mode:bool=False, idx:int=None, inst:Type['AOMEB']=None):
#        self.gamma = gamma
        if append_mode is True:
            self.idx = init_point_set[0]['idx']
            self.eps = eps

            self.radius2 = 0.0
            self.c_norm = 0.0
            
            self.kernel_cache = []
            self.core_points = []
            self.coefficents = []

            self.coreset_initial(init_point_set)
        else:
            self.idx = idx
            self.eps = inst.eps

            self.core_points = inst.core_points
            self.coefficents = inst.coefficents
            
            self.radius2 = inst.radius2
            
    def coreset_initial(self, points:[]) -> None:
#        init_coreset = KernelCoreMEB(points, self.eps, gamma=self.gamma)
        init_coreset = KernelCoreMEB(points, self.eps)

        self.radius2 = init_coreset.radius2
        self.c_norm = init_coreset.c_norm
        
        for idx in init_coreset.core_indices:
            self.core_points.append(init_coreset.points[idx])
            self.coefficents.append(init_coreset.coefficents[idx])
            
        self.init_kernel_matrix()
        
    def init_kernel_matrix(self):
        for i in range(len(self.core_points)):
            kernel_vector = []
            for j in range(len(self.core_points)):
                kernel_vector.append(self.rbf_eval(self.core_points[i], self.core_points[j]))
            self.kernel_cache.insert(i, kernel_vector)
            
    def append(self, points:[]) -> None:
        new_core_points = []
        
        for point in points:
            if self._dist2_wc(point) > (1.0 + self.eps) * (1.0 + self.eps) * self.radius2:
                new_core_points.append(point)
                
        if len(new_core_points) > 0:
            self.solve_apx_ball(new_core_points)
            
         
    def solve_apx_ball(self, points:[]) -> None:
        for point in points:
            self.core_points.append(point)
            self.coefficents.append(0.0)
            self.add_to_kernel_matrix(point)
            
        furthest_pair = self.find_farthest_point()
        nearest_pair = self.find_nearest_point()
        
        delta_plus = furthest_pair['dist2'] / (self.radius2 + 1e-20) - 1.0
        delta_minus = 1.0 - nearest_pair['dist2'] / (self.radius2 + 1e-20)
        delta = max(delta_plus, delta_minus)
        
        while delta > (1.0 + self.eps) * (1.0 + self.eps) - 1.0:
            if delta > delta_minus:
                _lambda = delta / (2.0 * (1.0 + delta))
                for i in range(len(self.core_points)):
                    if self.coefficents[i] >= 1e-12:
                        self.coefficents[i] = (1.0 - _lambda) * self.coefficents[i]
                        
                self.coefficents[furthest_pair['idx']] = self.coefficents[furthest_pair['idx']] + _lambda
            else:
                lambda1 = delta_minus / (2.0  * (1.0 - delta_minus))
                lambda2 = self.coefficents[nearest_pair['idx']] / (1.0 - self.coefficents[nearest_pair['idx']] + 1e-20)
                _lambda = min(lambda1, lambda2)
                
                for i in range(len(self.core_points)):
                    if self.coefficents[i] >= 1e-12:
                        self.coefficents[i] = (1.0 + _lambda) * self.coefficents[i]
                        
                self.coefficents[nearest_pair['idx']] = self.coefficents[nearest_pair['idx']] - _lambda
                
            self.update_c_norm()
            self.radius2 = 1.0 - self.c_norm
            
            furthest_pair = self.find_farthest_point()
            nearest_pair = self.find_nearest_point()
            
            delta_plus = furthest_pair['dist2'] / (self.radius2 + 1e-20) - 1.0
            delta_minus = 1.0 - nearest_pair['dist2'] / (self.radius2 + 1e-20)
            delta = max(delta_plus, delta_minus)
                
    def add_to_kernel_matrix(self, point:[]) -> None:
        kernel_vector = []
        for i in range(len(self.core_points) - 1):
            value = self.rbf_eval(point, self.core_points[i])
            kernel_vector.append(value)
            self.kernel_cache[i].append(value)
            
        kernel_vector.append(self.rbf_eval(point, point))
        self.kernel_cache.insert(len(self.core_points) - 1, kernel_vector)
    
    def find_nearest_point(self) -> {}:
        min_sq_dist = np.inf
        nearest_point = -1
        
        for i in range(len(self.core_points)):
            if self.coefficents[i] < 1e-12:
                continue
            
            sq_dist = self._dist2_wc_idx(i)
            
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                nearest_point = i
                
        return {"idx": nearest_point, "dist2": min_sq_dist}
    
    def find_farthest_point(self) -> {}:
        max_sq_dist = 0.0
        farthest_point = -1
        
        for i in range(len(self.core_points)):
            sq_dist = self._dist2_wc_idx(i)
            
            if sq_dist >= max_sq_dist:
                max_sq_dist = sq_dist
                farthest_point = i
                
        return {"idx": farthest_point, "dist2": max_sq_dist}
    
    def update_c_norm(self) -> None:
        self.c_norm = 0.0
        for i in range(len(self.core_points)):
            if self.coefficents[i] < 1e-12:
                continue
            for j in range(len(self.core_points)):
                if self.coefficents[j] < 1e-12:
                    continue
                self.c_norm += self.coefficents[i] * self.coefficents[j] * self.kernel_cache[i][j]
        
    def rbf_eval(self, p1:[], p2:[]) -> float:
        if p1['idx'] == p2['idx']:
            return 1.0
        else:
            return elm_kernel_vec(p1['data'], p2['data'])
            #return np.exp(-self._sq_dist(p1['data'], p2['data']) / self.gamma)

    def _sq_dist(self, x:[], y:[]) -> float:
        """Calculate squared euclidean distance between two points"""
        return np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)
    
    def _dist2_wc(self, point:[]) -> float:
        dist2 = 0.0
        for i in range(len(self.core_points)):
            if self.coefficents[i] < 1e-12:
                continue
            dist2 += self.coefficents[i] * self.rbf_eval(self.core_points[i], point)
        return 1.0 + self.c_norm - 2.0 * dist2
    
    def _dist2_wc_idx(self, idx1:int) -> float:
        dist2 = 0.0
        for idx2 in range(len(self.core_points)):
            if self.coefficents[idx2] < 1e-12:
                continue
            dist2 += self.coefficents[idx2] * self.kernel_cache[idx1][idx2]
        
        return 1.0 + self.c_norm - 2.0 * dist2