#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:57:40 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from libMulticlassFast.kernel_swmeb_plus import KernelSWMEB_Plus
import numpy as np
from libMulticlassFast.elm_kernel import elm_kernel_vec

class MCCVM:
    """ eps : float
            epsilon value for eps-approx meb
        w_size : int
            window size to be taken into swmeb
    """  

    def __init__(self, eps=0.1, w_size=100, kernel_fun=elm_kernel_vec):
        self.eps = eps
        self.w_size = w_size
        self.initial_fit = True
        self.kernel_fun = kernel_fun
        self.label_map = {}
        self.classifier = KernelSWMEB_Plus(eps1=self.eps, 
                                        window_size=self.w_size, batch_size=1)
    def partial_fit(self, X:np.array, y:np.array, classes=None):
        """ X : np.array
                data
            y : np.array
                labels
            classes : np.array
                unique class labels
        """
        # lets modify y to matrix
        
        if self.initial_fit is True:
            if classes is None:
                raise ValueError('Classes not provided on initial fit call')
                
            n_classes = len(classes)

            for cl, i in zip(classes, range(len(classes))):
                # create null vector
                self.label_map[cl] = np.zeros(n_classes)
                # set 1 at current label
                self.label_map[cl][i] = 1
                # create a label map like 3 classes label 1 = [0 1 0]
                # initialize a sliding window meb for each class

            self.initial_fit = False
            # store it once for prediction via argmax
            self.label_list = list(self.label_map.keys())
                    
        if (len(X.shape)) == 1:
            # single datapoint
            X = np.array([X])
            
        if X.shape[0] != y.size:
            raise ValueError(f'y has size {y.size} but X has {X.shape[0]} samples')
            
        for i in range(X.shape[0]):
            sample, lab = X[i], self.label_map[y[i]]

            # increment the sample count and append data to our MEB
            # we need to pass labels too
            self.classifier.append([sample], [lab])

    def predict(self, X:np.array, silent=False):
        if (len(X.shape)) == 1:
            # single datapoint
            X = np.array([X])
            
        y_pred = []
        
        # we need our coreset points with labels (or maybe better all window points? havent stored them)
        instance = self.classifier.get_active_instance()
        core_points = instance.core_points
        coefficients = instance.coefficents
                
        sim_per_class = []
        
#        print(core_points)
#        print('----\n')
        
        for x in X:
            # predict datapoint via eq 9
            # cache kernel for each core point
            kernel_cache = [(self.kernel_fun(core_points[i]['data'].reshape(1, -1), 
                                    x.reshape(1, -1)) + 1).ravel()[0] * coefficients[i] for i in range(len(core_points))]
            
            for label in self.label_map: 
#                sim = np.sum((self.kernel_fun(core_points[i]['data'].reshape(1, -1), 
#                                        x) + 1) * (coefficients[i] * (core_points[i]['label'] @ self.label_map[label])) for i in range(len(core_points)))
                sim = np.sum(kernel_cache[i] * (np.dot(core_points[i]['label'], self.label_map[label])) for i in range(len(core_points)))

                sim_per_class.append(sim)

            idx = np.argmax(sim_per_class)
                    
            y_pred.append(self.label_list[idx])
        
        return np.array(y_pred)
    
    def _dist(self, x:[], y:[]) -> float:
        """Calculate euclidean distance between two points"""
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    
    def get_info(self):
        return f'MCCVM eps: {self.eps}, w_size: {self.w_size} kernel: {self.kernel_fun}'