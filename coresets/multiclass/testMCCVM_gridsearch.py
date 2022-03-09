#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:17:30 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from mccvm import MCCVM
from skmultiflow.data import SEAGenerator, LEDGenerator, FileStream, MIXEDGenerator, AGRAWALGenerator, ConceptDriftStream
from sklearn.metrics import accuracy_score
from skmultiflow.lazy import SAMKNNClassifier as SAMKNN
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as ARSLVQ
from skmultiflow.meta import OnlineUnderOverBaggingClassifier
from skmultiflow.bayes import NaiveBayes
from libSAM.elm_kernel import elm_kernel_vec
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity, laplacian_kernel, rbf_kernel, sigmoid_kernel
import json
import time
import copy
import random
import os
import numpy as np

try:
    # clean up old res
    os.remove('res_grid_mccvm.json')
except:
    pass

clfs = [ARSLVQ(gradient_descent='adadelta')]

#for w_size in [5, 50, 100, 300, 1000]:
for w_size in [300, 1000, 50]:
    for eps in [0.01, 0.05, 0.1]:
        for kernel_fun in [linear_kernel, cosine_similarity, laplacian_kernel, rbf_kernel, sigmoid_kernel, elm_kernel_vec]:
            clfs.append(MCCVM(w_size=w_size, kernel_fun=kernel_fun, eps=eps))

PRETRAIN_SIZE:int = 100
N_SAMPLES:int = 300000
REJECTION_LABEL = 0
REJECTION_CHANCE = 0

first = True

for _clf in clfs:
    streams = [
                ConceptDriftStream(MIXEDGenerator(classification_function=0), MIXEDGenerator(classification_function=1), position=N_SAMPLES // 2, width=1), 
                ConceptDriftStream(SEAGenerator(classification_function=0), SEAGenerator(classification_function=1), position=N_SAMPLES // 2, width=1), 
                ConceptDriftStream(MIXEDGenerator(classification_function=0), MIXEDGenerator(classification_function=1), position=N_SAMPLES // 2, width=N_SAMPLES // 20), 
                ConceptDriftStream(SEAGenerator(classification_function=0), SEAGenerator(classification_function=1), position=N_SAMPLES // 2, width=N_SAMPLES // 20), 
                LEDGenerator(has_noise=True, noise_percentage=0.1),
                AGRAWALGenerator(), 
                FileStream('data/weather.csv'),
                FileStream('data/elec.csv'),
                FileStream('data/gmsc.csv'),
                FileStream('data/poker.csv'),
                FileStream('data/moving_squares.csv')
                ]
    
    for stream in streams:
        # reset our scores
        y_true = []
        y_pred = []
        
        # clone the classifier
        clf = copy.deepcopy(_clf)

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
                
            if (stream.current_sample_y[0] == REJECTION_LABEL) and random.random() < REJECTION_CHANCE:
                continue
                
            x, y = stream.current_sample_x, stream.current_sample_y
            
            # sometimes labels are additonally wrapped like multiflows CD Stream
            if isinstance(y, (np.ndarray)):
                y = y.flatten()
            
            y_pred.append(clf.predict(x))
            y_true.append(y)
            clf.partial_fit(x, y, classes=stream.target_values)
            
        end_time = time.time() - start_time
        
        print(stream)
        print(clf.get_info())
        print(accuracy_score(y_pred, y_true))

        f = open('res_grid_mccvm.json', 'a')

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
        f.write(f'\t"acc": {accuracy_score(y_pred, y_true)},\n')
        f.write(f'\t"time": {end_time}\n')
        f.write('}')
        f.close()

f = open('res_grid_mccvm.json', 'a')
f.write(']')
f.close()

# TODO iterate over file and find best acc of sw clf (gridsearch)
f = open('res_grid_mccvm.json', 'r')
data = f.read()
data_json = json.loads(data)
f.close()

try:
    # clean up old best
    os.remove('best_grid_mccvm.json')
except:
    pass

classifiers = []

first = True

for entry in data_json:
    # for clf/stream combination
    classifier = entry['clf'].split(' ')[0]
    stream = entry['stream']
    
    if str(classifier) + str(stream) in classifiers:
        continue # already processed
        
    classifiers.append(str(classifier) + str(stream))
    
    best_acc = 0
    
    # for stream in stream
    for _entry in data_json:
        _classifier = _entry['clf'].split(' ')[0]
        
        # ensure its the wanted stream and classifier
        if classifier == _classifier and stream == _entry['stream']:
            if _entry['acc'] >= best_acc:
                best_acc = _entry['acc']
                best = _entry

    f = open('best_grid_mccvm.json', 'a')
    if first is False:
        f.write(',\n')
    else:
        first = False
        
    f.write('{\n')
    f.write(f'\t"stream": "{best["stream"]}"'.replace('\n', ''))
    f.write(',\n')
    f.write(f'\t"clf": "{best["clf"]}"'.replace('\n', ''))
    f.write(',\n')
    f.write(f'\t"acc": {best["acc"]},\n')
    f.write(f'\t"time": {best["time"]}\n')
    f.write('}')
    f.close()
