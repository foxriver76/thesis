#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 07:56:23 2020

@author: moritz
"""

from .aomeb import AOMEB
import numpy as np

class SWMEB_Plus:
    
    def __init__(self, eps1=0.1, window_size=1000, batch_size=10):
        self.cur_id = -1
        self.eps1 = eps1
        self.W = window_size
        self.BATCH_SIZE = batch_size
        self.EPS_MAX = 0.1
        self.EPS_MIN = 1e-6
        self.LAMBDA = 4.0
        self._n = 0
        
        self.index = []
        self.instances = {}
        
    def append(self, point_set:[]) -> None:
        for i in range(len(point_set)):
            point_set[i] = {"idx": self._n, "data": point_set[i]}
            self._n += 1
            
        self.cur_id = point_set[self.BATCH_SIZE - 1]['idx']
        
        # TODO:(check) instance removed so compress data of previous instance and add it to the currect one
        while len(self.index) > 2 and self.index[1] <= self.cur_id - self.W:
            rmInst = self.instances.pop(self.index.pop(0))
            #print(f'instance removed:\n{rmInst.core_points}')
            all_points = []
            for point in rmInst.core_points:
                all_points.append(point['data'])
                
            preserve_point = np.mean(all_points, axis=0)
            self.append([preserve_point])
                
            
        if len(self.instances) > 0:
            # if there are existing instances, add point set to all of them
            for inst in self.instances:
                self.instances[inst].append(point_set)
                
        new_inst = AOMEB(init_point_set=point_set, eps=self.eps1, append_mode=True)
        
        self.index.append(new_inst.idx)
        self.instances[new_inst.idx] = new_inst
        
        to_delete = []
        cur = 0
        beta = self.EPS_MIN
        
        while cur < len(self.index) - 2:
            pre = cur
            cur = self.find_next(cur, beta)
            
            if cur - pre > 1:
                i = pre + 1
                while i < cur:
                    to_delete.append(self.index[i])
                    i += 1
                
            beta = beta * self.LAMBDA
            beta = min(beta, self.EPS_MAX)
            
        for del_id in to_delete:
            self.index.remove(del_id)
            self.instances.pop(del_id)
            
    def find_next(self, cur:int, beta:float) -> int:
        cur_radius = self.instances[self.index[cur]].radius
        nxt_radius = self.instances[self.index[cur + 1]].radius
        
        if cur_radius / (nxt_radius + 1e-12) >= 1.0 + beta:
            nxt = cur + 1
        else:
            i = cur + 2
            nxt_radius = self.instances[self.index[i]].radius
            while i < len(self.index) - 1 and cur_radius / (nxt_radius + 1e-12) <= 1.0 + beta:
                i += 1
                nxt_radius = self.instances[self.index[i]].radius
                
            if i == len(self.index) - 1 and cur_radius / (nxt_radius + 1e-12) <= 1.0 + beta:
                nxt = i
            else: 
                nxt = i - 1
                
        return nxt
    
    def approx_meb(self) -> None:
        if self.index[0] >= self.cur_id - self.W + 1 and self.cur_id - self.W + 1 > 0:
            self.instances[self.index[1]].approx_meb()
        else:
            self.instances[self.index[0]].approx_meb()