#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:05:57 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from sklearn.metrics import euclidean_distances

class StreamMEB():
    """Stream Minimum Enclosing Ball
    -----------------------
    Zarrabi-Zadeh, Hamid and Timothy M. Chan. 
    “A Simple Streaming Algorithm for Minimum Enclosing Balls.” CCCG (2006).
    """
    
    def __init__(self):
        self.radius = 0
        self.center = 0
        self.fitted = False
        # Support vectors first array for DP and second for index of Stream, to check how old it is
        self.support_vectors = [[], []]
        
    def partial_fit(self, p_i, iteration, removeIteration=None):
        """Fit one new datapoint"""
        if self.fitted is False:
            """Do inital fit"""
            self.center = p_i
            self.radius = 0
            self.fitted = True
        else:
            """Check if DP is outside of ball"""
            distance = euclidean_distances(p_i.reshape(1, p_i.size), self.center.reshape(1, p_i.size))
            if distance > self.radius:
                """DP is outside of ball -> calc new ball"""
                # Delta is half the distance between p_i and B_i-1
                delta_i = 0.5 * (distance - self.radius)
                # Calc r_i
                self.radius = self.radius + delta_i
                # Calc c_i
                self.center = self.center + (delta_i / distance) * (p_i - self.center)
                if iteration is not None:
                    # Save Support Vector and Iteration
                    self.support_vectors[0].append(p_i)
                    self.support_vectors[1].append(iteration)
                    if removeIteration is not None:
                        self.remove_old_vectors(removeIteration)

    def get_info(self):
        return 'StreamMEB Info:\nCenter: {}\nRadius: {}\nSupport Vectors: {}\nSupport Vectors Iteration: {}'.format(
                self.center, self.radius, self.support_vectors[0], self.support_vectors[1])
        
    def remove_old_vectors(self, iteration):
        """Removes old vectors before iteration and calculates new ball out of stored vectors"""
        updated = False
            
        while 3 < len(self.support_vectors[1]):
            # Store at least three points
            if self.support_vectors[1][0] < iteration:
                self.support_vectors[1].pop(0)
                self.support_vectors[0].pop(0)
                updated = True
            else:
                # break because higher index will correspond to higher iteration
                break
            
        if updated is True:
            """Support Vectors have been removed --> calc new MEB"""
            self.fitted = False
            
            print('Update current ball: C: {} R: {}'.format(self.center, self.radius))
            for i in range(len(self.support_vectors[0])):
                self.partial_fit(self.support_vectors[0][i], iteration=None)
            print('Got new ball: C: {} R: {}'.format(self.center, self.radius))
