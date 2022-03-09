#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:57:40 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from libSAM.swmeb_plus import SWMEB_Plus
from libSAM.kernel_swmeb_plus import KernelSWMEB_Plus
import numpy as np
from libSAM.elm_kernel import elm_kernel_vec

class SWMEBClf:
    """ eps : float
            epsilon value for eps-approx meb
        w_size : int
            window size to be taken into swmeb
    """  

    def __init__(self, eps=0.1, w_size=100, kernelized=False, only_misclassified=False, kernel_fun=elm_kernel_vec):
        self.eps = eps
        self.w_size = w_size
        self.swmeb = KernelSWMEB_Plus if kernelized else SWMEB_Plus
        self.initial_fit = True
        self.classifiers = {}
        self.kernelized = kernelized
        self.only_misclassified = only_misclassified
        self.kernel_fun = kernel_fun
    
    def partial_fit(self, X:np.array, y:np.array, classes:np.array):
        """ X : np.array
                data
            y : np.array
                labels
            classes : np.array
                unique class labels
        """
        if self.initial_fit is True:
            for cl in classes:
                # initialize a sliding window meb for each class
                self.classifiers[cl] = self.swmeb(eps1=self.eps, 
                                        window_size=self.w_size, batch_size=1)
            
            self.initial_fit = False
                    
        if (len(X.shape)) == 1:
            # single datapoint
            X = np.array([X])
            
        if X.shape[0] != y.size:
            raise ValueError(f'y has size {y.size} but X has {X.shape[0]} samples')
            
        for i in range(X.shape[0]):
            sample, lab = X[i], y[i]
            misclassified = True
            
            if self.only_misclassified is True:
                pred = self.predict(sample, silent=True)
                misclassified = not (pred[0] == lab)

            if misclassified is True:
                # increment the sample count and append data to our MEB
                self.classifiers[lab].append([sample])
            else:
                #print('no learn')
                pass
            
    def predict(self, X:np.array, silent=False):
        if (len(X.shape)) == 1:
            # single datapoint
            X = np.array([X])
            
        y_pred = []
        
        for x in X:
            y = None
            minDist = None
            
            for label in self.classifiers:
                try:
                    model = self.classifiers[label].instances[self.classifiers[label].index[0]]
                except Exception as e:
                    if silent is None:
                        print(f'Skipping classifier for label {label}: {e}')
                        print(self.classifiers[label].instances)
                    continue
                
                # first calculate dist to center
                if self.kernelized:
                    # idx -1 to have no matching idx
                    dist = model._dist2_wc({"idx": -1, "data": x})
                else:
                    dist = self._dist(x, model.center)
                    
                # then calculate dist only to the outside of the ball
                if self.kernelized:
                    dist -= model.radius2
                else:
                    dist -= model.radius

                if minDist is None or dist < minDist:
                    minDist = dist
                    y = label
                    
            y_pred.append(y)
        
        return np.array(y_pred)
    
    def _dist(self, x:[], y:[]) -> float:
        """Calculate euclidean distance between two points"""
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))
    
    def get_info(self):
        return f'SWMEB eps: {self.eps}, kernelized: {self.kernelized}, w_size: {self.w_size} only_mis: {self.only_misclassified} kernel: {self.kernel_fun}'