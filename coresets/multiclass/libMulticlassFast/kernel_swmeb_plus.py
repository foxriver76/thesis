#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 08:36:02 2020

@author: moritz
"""

from .kernel_aomeb import Kernel_AOMEB
from .elm_kernel import elm_kernel_vec

class KernelSWMEB_Plus:
    
    def __init__(self, eps1:float=0.1, window_size:int=1000, batch_size:int=10, kernel_fun=elm_kernel_vec):
        self.cur_id = -1
        self.eps1 = eps1
        self.W = window_size
        self.BATCH_SIZE = batch_size
        self.EPS_MAX = 0.1
        self.EPS_MIN = 1e-6
        self.LAMBDA = 4.0
        self._n = 0
        self.kernel_fun = kernel_fun

#        self.gamma = gamma
        
        self.index = []
        self.instances = {}
      
    def append(self, point_set:[], labels) -> None:
        for i in range(len(point_set)):
            point_set[i] = {"idx": self._n, "data": point_set[i], "label": labels[i]}
            self._n += 1
           
        self.cur_id = point_set[self.BATCH_SIZE - 1]['idx']
        # delete outdated instances until there is only one outdated
        while len(self.index) > 2 and self.index[1] <= self.cur_id - self.W:
            self.instances.pop(self.index.pop(0))
            
        if len(self.instances) > 0:
            #for inst in self.instances:
            for inst in self.index:
                self.instances[inst].append(point_set)
                
        # create a new aomeb with the new points
        new_inst = Kernel_AOMEB(init_point_set=point_set, eps=self.eps1, kernel_fun=self.kernel_fun)

        self.index.append(new_inst.idx)
        self.instances[new_inst.idx] = new_inst
                
        # delete instances which can be approxiamted by the succesors     
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
        cur_radius2 = self.instances[self.index[cur]].radius2
        nxt_radius2 = self.instances[self.index[cur + 1]].radius2
        
        if cur_radius2 / (nxt_radius2 + 1e-20) >= (1.0 + beta) * (1.0 + beta):
            nxt = cur + 1
        else:
            i = cur + 2
            nxt_radius2 = self.instances[self.index[i]].radius2
            while i < len(self.index) - 1 and cur_radius2 / (nxt_radius2 + 1e-20) <= (1.0 + beta) * (1.0 + beta):
                i += 1
                nxt_radius2 = self.instances[self.index[i]].radius2
                
            if i == len(self.index) - 1 and cur_radius2 / (nxt_radius2 + 1e-20) <= (1.0 + beta) * (1.0 + beta):
                nxt = i
            else: 
                nxt = i - 1
                
        return nxt
    
    def get_active_instance(self) -> Kernel_AOMEB:
        # if instance at 0 outdated, return instance at 1 (we only can have one outdated)
        # TODO: https://github.com/yhwang1990/SW-MEB/issues/1
        if self.index[0] < self.cur_id - self.W + 1:
            return self.instances[self.index[1]]
        else:
            return self.instances[self.index[0]]
    
    def approx_meb(self) -> None:
        # if index 0 outdated, use 1
        # TODO: https://github.com/yhwang1990/SW-MEB/issues/1
        if self.index[0] < self.cur_id - self.W + 1:
            self.instances[self.index[1]].approx_meb()
        else:
            self.instances[self.index[0]].approx_meb()