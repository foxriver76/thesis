#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:54:01 2020

@author: moritz
"""
from typing import Type
from .core_meb import CoreMEB
import numpy as np

class AOMEB:
    
    def __init__(self, init_point_set:[]=None, eps:float=None, append_mode:bool=False, idx:int=None, inst:Type['AOMEB']=None):
        if append_mode is True:
            self.idx = init_point_set[0]['idx']
            self.eps = eps
            self.core_points = []
            self.center = np.zeros_like(init_point_set[0]['data'])
            self.radius = 0.0
            
            self.coreset_construct(init_point_set)
        else:
            self.idx = idx
            self.eps = inst.eps
            self.core_points = inst.core_points
            self.center = inst.center
            self.radius = inst.radius
            
    def coreset_construct(self, points:[]) -> None:
        first_point = points[0]
        p1 = self.find_farthest_point(points=points, p=first_point)
        p2 = self.find_farthest_point(points=points, p=p1)
        
        self.radius = self._dist(p1['data'], p2['data']) / 2.0
        self.center = (p1['data'] + p2['data']) / 2.0
                
        self.core_points.insert(0, p1)
        
        # but if its the same point, then only add one
        if p1['idx'] != p2['idx']:
            self.core_points.insert(0, p2)
        
        while True:
            furthest_point = self.find_farthest_point(points)
            max_dist = self._dist(self.center, furthest_point['data'])

            if max_dist <= self.radius * (1.0 + self.eps): break
        
            self.radius = (self.radius * self.radius / max_dist + max_dist) / 2.0
            self.center = furthest_point['data'] + (self.radius / max_dist) * (self.center - furthest_point['data'])

            self.core_points.insert(0, furthest_point)
            self.solve_apx_ball()
            
    def solve_apx_ball(self) -> None:
        while True:
            furthest_point = self.find_farthest_point(self.core_points)
            max_dist = self._dist(self.center, furthest_point['data'])
            if max_dist <= self.radius * (1.0 + self.eps / 2.0): break
            
            self.radius = (self.radius * self.radius / max_dist + max_dist) / 2.0
            self.center = furthest_point['data'] + (self.radius / max_dist) * (self.center - furthest_point['data'])
    
    def find_farthest_point(self, points:[], p:{}=None) -> {}:
        max_sq_dist = -1.0 # init negative to be smaller than zero
        farthest_point = None
        
        dist_point = self.center if p is None else p['data']
        
        for point in points:
            sq_dist = self._sq_dist(dist_point, point['data'])
            if sq_dist >  max_sq_dist:
                max_sq_dist = sq_dist
                farthest_point = point
            
        return farthest_point
    
    def append(self, points: []) -> None:
        new_core_points = []
        for point in points:
            # if point outside ball, this point is a new core point
            if self._dist(point['data'], self.center) > (1.0 + self.eps) * self.radius:
                new_core_points.insert(0, point)
                
        if len(new_core_points) > 0:
            self.core_points.extend(new_core_points)
            self.solve_apx_ball()
            
    def approx_meb(self):
        coreset = CoreMEB(self.core_points, self.eps)
        self.center = coreset.center
        self.radius = coreset.radius      
        
    def _dist(self, x:[], y:[]) -> float:
        """Calculate euclidean distance between two points"""
        dis = np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
        # somehow python has sometimes problem and returns nan when self distance is needed
        if np.isnan(dis):
            return 0

        return dis
    
    def _sq_dist(self, x:[], y:[]) -> float:
        """Calculate squared euclidean distance between two points"""
        return np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y)

        
        