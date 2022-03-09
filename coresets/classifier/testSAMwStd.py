#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:17:30 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import sys
import os
sys.path.append(os.path.abspath(__file__ + "/../../"))
from meb_classifier_sam import SWMEBClf
from skmultiflow.data import SEAGenerator, LEDGenerator, FileStream, MIXEDGenerator, AGRAWALGenerator, ConceptDriftStream
from sklearn.metrics import accuracy_score
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as ARSLVQ
from libSAM.elm_kernel import elm_kernel_vec
from sklearn.metrics.pairwise import linear_kernel
import json
import time
import copy
import numpy as np

try:
    # clean up old res
    os.remove('res_w_std.json')
except:
    pass

# optimized clfs
clfs = [ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        ARSLVQ(gradient_descent='adadelta'),
        SWMEBClf(eps=0.1, w_size=300, kernelized=True, only_misclassified=False, kernel_fun=linear_kernel),
        SWMEBClf(eps=0.1, w_size=50, kernelized=False, only_misclassified=False, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=100, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=100, kernelized=False, only_misclassified=False, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=300, kernelized=False, only_misclassified=False, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=300, kernelized=False, only_misclassified=True, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=5, kernelized=True, only_misclassified=True, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=5, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=5, kernelized=False, only_misclassified=True, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=5, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec),
        SWMEBClf(eps=0.1, w_size=5, kernelized=True, only_misclassified=False, kernel_fun=elm_kernel_vec),
        ]

PRETRAIN_SIZE:int = 100
N_SAMPLES:int = 300000

streams = [
            ConceptDriftStream(MIXEDGenerator(classification_function=0), MIXEDGenerator(classification_function=1), position=N_SAMPLES // 2, width=1), 
            ConceptDriftStream(SEAGenerator(classification_function=0), SEAGenerator(classification_function=1), position=N_SAMPLES // 2, width=1), 
            ConceptDriftStream(MIXEDGenerator(classification_function=0), MIXEDGenerator(classification_function=1), position=N_SAMPLES // 2, width=N_SAMPLES // 20), 
            ConceptDriftStream(SEAGenerator(classification_function=0), SEAGenerator(classification_function=1), position=N_SAMPLES // 2, width=N_SAMPLES // 20), 
            LEDGenerator(has_noise=True, noise_percentage=0.1),
            AGRAWALGenerator(), 
            FileStream('../../datasets/weather.csv'),
            FileStream('../../datasets/elec.csv'),
            FileStream('../../datasets/gmsc.csv'),
            FileStream('../../datasets/poker.csv'),
            FileStream('../../datasets/moving_squares.csv'),
            ConceptDriftStream(MIXEDGenerator(classification_function=0), MIXEDGenerator(classification_function=1), position=N_SAMPLES // 2, width=1), 
            ConceptDriftStream(SEAGenerator(classification_function=0), SEAGenerator(classification_function=1), position=N_SAMPLES // 2, width=1), 
            ConceptDriftStream(MIXEDGenerator(classification_function=0), MIXEDGenerator(classification_function=1), position=N_SAMPLES // 2, width=N_SAMPLES // 20), 
            ConceptDriftStream(SEAGenerator(classification_function=0), SEAGenerator(classification_function=1), position=N_SAMPLES // 2, width=N_SAMPLES // 20), 
            LEDGenerator(has_noise=True, noise_percentage=0.1),
            AGRAWALGenerator(), 
            FileStream('../../datasets/weather.csv'),
            FileStream('../../datasets/elec.csv'),
            FileStream('../../datasets/gmsc.csv'),
            FileStream('../../datasets/poker.csv'),
            FileStream('../../datasets/moving_squares.csv')
            ]



first = True


    
for ii in range(len(streams)):    
    # 5 fold
    fold_acc = []
    fold_time = []
    for fold in range(5):
        clf = copy.deepcopy(clfs[ii])
        stream = copy.deepcopy(streams[ii])
        # reset our scores
        y_true = []
        y_pred = []
    
        start_time = time.time()
    
        stream.next_sample(PRETRAIN_SIZE)
        
        x, y = stream.current_sample_x, stream.current_sample_y
        
        # sometimes labels are additonally wrapped like multiflows CD Stream
        if isinstance(y, (np.ndarray)):
            y = y.flatten()
        
        clf.partial_fit(x, y, classes=stream.target_values)
        
        for i in range(N_SAMPLES):
            stream.next_sample()
            
            if stream.has_more_samples() is False:
                break
                
            x, y = stream.current_sample_x, stream.current_sample_y
            
            # sometimes labels are additonally wrapped like multiflows CD Stream
            if isinstance(y, (np.ndarray)):
                y = y.flatten()
            
            y_pred.append(clf.predict(x))
            y_true.append(y)
            clf.partial_fit(x, y, classes=stream.target_values)
            
        end_time = time.time() - start_time
        
        fold_acc.append(accuracy_score(y_pred, y_true))
        fold_time.append(end_time)
        
    
    f = open('res_w_std.json', 'a')

    if first is False:
        f.write(',\n')
    else:
        f.write('[')     
        first = False
        
    f.write('{\n')
    f.write(f'\t"stream": "{stream}"'.replace('\n', ''))
    f.write(',\n')
    f.write(f'\t"clf": "{clf.get_info()}"'.replace('\n', ''))
    f.write(',\n')
    f.write(f'\t"acc": {np.mean(fold_acc)} +- {np.std(fold_acc)},\n')
    f.write(f'\t"time": {np.mean(fold_time)} +- {np.std(fold_time)}\n')
    f.write('}')
    f.close()

f = open('res_w_std.json', 'a')
f.write(']')
f.close()

    