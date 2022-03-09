#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:08:17 2020

@author: moritz
"""

import numpy as np

class CoreMEB:
    
    def __init__(self, point_set:[], eps:float):
        self.core_points = []
        self.center = None
        self.radius = 0.0
        self.eps = eps
        
        self.coreset_construct(point_set)
        
    def coreset_construct(self, points:[]) -> None:
        first_point = points[0]
        p1 = self.find_farthest_point(points=points, p=first_point)
        p2 = self.find_farthest_point(points=points, p=p1)
        
        self.radius = self._dist(p1['data'], p2['data']) / 2.0
        self.center = (np.array(p1['data'])  + np.array(p2['data'])) / 2.0
        
        self.core_points.insert(0, p1)
        self.core_points.insert(0, p2)
        
        while True:
            furthest_point = self.find_farthest_point(points)
            max_dist = self._dist(self.center, furthest_point['data'])
            
            if max_dist <= self.radius * (1.0 + self.eps): break
        
            self.radius = (self.radius * self.radius / max_dist + max_dist) / 2.0
            self.center = (furthest_point['data'] + (self.radius / max_dist) * (self.center - furthest_point['data'])).ravel()
            
            self.core_points.insert(0, furthest_point)
            
            self.solve_apx_ball()
            
    def find_farthest_point(self, points:[], p:{}=None) -> {}:
        max_sq_dist = 0.0
        farthest_point = None
        
        dist_point = self.center if p is None else p['data']
        
        for point in points:
            sq_dist = self._sq_dist(dist_point, point['data'])
            if sq_dist >  max_sq_dist:
                max_sq_dist = sq_dist
                farthest_point = point
        
        return farthest_point
    
    def solve_apx_ball(self) -> None:
        while True:
            furthestPoint = self.find_farthest_point(self.core_points)
            max_dist = self._dist(self.center, furthestPoint['data'])
            
            if max_dist <= self.radius * (1.0 + self.eps / 2.0): break
            
            self.radius = (self.radius * self.radius / max_dist + max_dist) / 2.0
            self.center = (furthestPoint['data'] + (self.radius / max_dist) * (self.center - furthestPoint['data'])).ravel()
            
    def _dist(self, x:[], y:[]) -> float:
        """Calculate euclidean distance between two points"""
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    
    def _sq_dist(self, x:[], y:[]) -> float:
        """Calculate squared euclidean distance between two points"""
        return np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)