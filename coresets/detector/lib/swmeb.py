#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:29:49 2020

@author: moritz
"""

from aomeb import AOMEB

class SWMEB:
    
    def __init__(self, eps1, window_size=1000, batch_size=10):
        self.cur_id = -1
        self.eps1 = eps1
        self.W = window_size
        
        self.CHUNK_SIZE = self.W / 10

        self.EPS_MAX = 0.1
        self.MIN_INST = 10
        self.BATCH_SIZE = batch_size
        
        self.buffer = []
        self.instances = []
        
    def append(self, point_set:[]) -> None:
        cur_id = point_set[self.BATCH_SIZE - 1]['idx']
        if len(self.instances) > 0 and self.instances[0].idx <= cur_id - self.W:
            # REMOVE first instance if instance is older than W
            self.instances.pop(0)
            
        if len(self.instances) > 0:
            for inst in self.instances:
                inst.append(point_set)
               
        # add whole point set to buffer
        self.buffer.extend(point_set)
        
        # if buffer larger chunk size, add instances and clear buffer
        if len(self.buffer) >= self.CHUNK_SIZE:
            self.add_instances()
            self.buffer.clear()
            
    def add_instances(self) -> None:
        new_inst = []
        init_batch_id = len(self.buffer) - self.BATCH_SIZE
        last_inst_idx = init_batch_id
        base_instance =  AOMEB(init_point_set=self.buffer[init_batch_id:init_batch_id + self.BATCH_SIZE], eps=self.eps1, append_mode=True)
        
        beta = self.EPS_MAX
        cur_radius = base_instance.radius
        
        # insert updated instance at index 0
        new_inst.insert(0, AOMEB(idx=base_instance.idx, inst=base_instance))
        
        for batch_id in reversed(range(0, int(self.CHUNK_SIZE // self.BATCH_SIZE - 2 - 1))):
            cur_batch_id = batch_id * self.BATCH_SIZE
            base_instance.append(self.buffer[cur_batch_id:cur_batch_id + self.BATCH_SIZE])
            
            if base_instance.radius / (cur_radius + 1e-12) >= 1.0 + beta:
                new_inst.insert(0, AOMEB(idx=self.buffer[cur_batch_id]['idx'], inst=base_instance))
                cur_radius = base_instance.radius
                last_inst_idx = cur_batch_id
            elif (last_inst_idx - cur_batch_id) >= (len(self.buffer) / self.MIN_INST):
                new_inst.insert(0, AOMEB(idx=self.buffer[cur_batch_id]['idx'], inst=base_instance))
                last_inst_idx = cur_batch_id

        for inst in new_inst:
            self.instances.append(inst)
            
    def approx_meb(self):
        self.instances[0].approx_meb()