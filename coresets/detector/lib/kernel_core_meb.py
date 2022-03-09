#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 07:48:29 2020

@author: moritz
"""

import numpy as np
from .elm_kernel import elm_kernel_vec

class KernelCoreMEB:
    
#    def __init__(self, point_set:[], eps:float, gamma:float=0.1):
    def __init__(self, point_set:[], eps:float):
        self.points = point_set
        self.core_indices = []
        self.coefficents = {}
        self.radius = 0.0
        
        self.c_norm = 0.0
        self.kernel_cache = {}
        
        self.eps = eps
#        self.gamma = gamma
        
        self.coreset_construct_with_away_steps()
        
    def coreset_construct_with_away_steps(self) -> None:
        first_point = self.points[0]

        pair1 = self.find_farthest_point(first_point)
        pair2 = self.find_farthest_point(self.points[pair1['idx']])
        
        # first two support vectors
        self.core_indices.append(pair1['idx'])
        self.core_indices.append(pair2['idx'])
        
        # both are equally relevant
        self.coefficents[pair1['idx']] = self.coefficents[pair2['idx']] = 0.5
        
        # calc rbf based kernel matrix of both protos
        self.init_kernel_matrix()
        
        # calc norm of our lagrangian coefficients
        self.update_c_norm()

        # calc squared radius
        self.radius2 = 1.0 - self.c_norm
        
        furthest_pair = self.find_farthest_point()
        nearest_pair = self.find_nearest_point()
        
        # if we only have one starting point - delta_minus and delta_plus are both zero
        if furthest_pair['idx'] == nearest_pair['idx']:
            delta = 0
        else:
            # calc max delta from our ''center''
            delta_plus = furthest_pair['dist2'] / (self.radius2 - 1.0)
            delta_minus = 1.0 - nearest_pair['dist2'] / (self.radius2)
            delta = max(delta_minus, delta_plus)
                
        while delta > (1.0 + self.eps) * (1.0 + self.eps) - 1.0:
            if delta > delta_minus:
                if furthest_pair['idx'] not in self.core_indices:
                    self.core_indices.append(furthest_pair['idx'])
                    self.add_to_kernel_matrix(furthest_pair['idx'])
                    
                _lambda = delta / (2.0 * (1.0 + delta))
                
                for idx in self.core_indices:
                    self.coefficents[idx] = (1.0 - _lambda) * self.coefficents[idx]
                    
                self.coefficents[furthest_pair['idx']] += _lambda      
            else:
                lambda1 = delta_minus / (2.0  * (1.0 - delta_minus))
                lambda2 = self.coefficents[nearest_pair['idx']] / (1.0 - self.coefficents[nearest_pair['idx']])
                _lambda = min(lambda1, lambda2)
                
                for idx in self.core_indices:
                    self.coefficents[idx] = (1.0 + _lambda) * self.coefficents[idx]
                    
                self.coefficents[nearest_pair['idx']] -= _lambda
                
                if self.coefficents[nearest_pair['idx']] <= 1e-12:
                    self.core_indices.remove(nearest_pair['idx'])
                    self.delete_from_kernel_matrix(nearest_pair['idx'])
            
            self.update_c_norm()
            self.radius2 = 1.0 - self.c_norm
            
            furthest_pair = self.find_farthest_point()
            nearest_pair = self.find_nearest_point()
            
            delta_plus = furthest_pair['dist2'] / self.radius2 - 1.0
            delta_minus = 1.0 - nearest_pair['dist2'] / self.radius2
            delta = max(delta_plus, delta_minus)
            
    def delete_from_kernel_matrix(self, idx:int) -> None:
        del self.kernel_cache[idx]
                
    def find_nearest_point(self) -> {}:
        min_sq_dist = np.inf
        nearest_point = -1
        
        for idx in self.core_indices:            
            sq_dist = self._dist2_wc_idx(idx)
            
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                nearest_point = idx
                
        return {"idx": nearest_point, "dist2": min_sq_dist}
    
    def find_farthest_point(self, point:[]=None) -> {}:
        max_sq_dist = 0.0
        farthest_point = -1
        
        if point is None:
            for i in range(len(self.points)):
                sq_dist = self._dist2_wc_idx(i)
                
                if sq_dist >= max_sq_dist:
                    max_sq_dist = sq_dist
                    farthest_point = i
        else:
            # we got a point
            for i in range(len(self.points)):
                sq_dist = self._k_dist2(point, self.points[i])
                
                if sq_dist >= max_sq_dist:
                    max_sq_dist = sq_dist
                    farthest_point = i
                
        return {"idx": farthest_point, "dist2": max_sq_dist}
    
    def update_c_norm(self) -> None:
        self.c_norm = 0.0
        for idx1 in self.core_indices:
            for idx2 in self.core_indices:
                self.c_norm += (self.coefficents[idx1] * self.coefficents[idx2] * self.kernel_cache[idx1][idx2])
                
    def add_to_kernel_matrix(self, idx:int) -> None:
        kernel_vector = []
        for i in range(len(self.points)):
            kernel_vector.append(self.rbf_eval(self.points[idx], self.points[i]))

        self.kernel_cache[idx] = kernel_vector
                
    def init_kernel_matrix(self):
        for idx in self.core_indices:
            kernel_vector = []
            for i in range(len(self.points)):
                kernel_vector.append(self.rbf_eval(self.points[idx], self.points[i]))
            self.kernel_cache[idx] = kernel_vector
            
    def _dist2_wc(self, point:[]) -> float:
        dist2 = 0.0
        for idx in self.core_indices:
            dist2 += self.coefficents[idx] * self.rbf_eval(self.core_points[idx], point)
        
        return 1.0 + self.c_norm - 2.0 * dist2
    
    def _dist2_wc_idx(self, idx1:int) -> float:
        dist2 = 0.0
        for idx2 in self.core_indices:
            if self.coefficents[idx2] < 1e-12:
                continue
            dist2 += self.coefficents[idx2] * self.kernel_cache[idx2][idx1]
        
        return 1.0 + self.c_norm - 2.0 * dist2
    
    def _k_dist2(self, p1:[], p2: []) -> float:
        return (2.0 - 2.0 * self.rbf_eval(p1, p2))
    
    def rbf_eval(self, p1:[], p2:[]) -> float:
        if p1['idx'] == p2['idx']:
            return 1.0
        else:
            return elm_kernel_vec(p1['data'], p2['data'])
#            return np.exp(- self._sq_dist(p1['data'], p2['data']) / self.gamma)
        
    def _sq_dist(self, x:[], y:[]) -> float:
        """Calculate squared euclidean distance between two points"""
        return np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)