# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:17:37 2019

@author: moritz
"""

import numpy as np
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as RSLVQ
from skmultiflow.trees import HAT, HoeffdingTree as HT
from skmultiflow.meta import AdaptiveRandomForest as ARF
from skmultiflow.lazy import SAMKNN
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.random_projection import SparseRandomProjection
from random_projection.model.arslvq import RSLVQ as ARSLVQ
import copy
import time

def next_sample():
    line = f.readline()
    if line == '': return 'EOF', 'EOF'
    arr = np.array(line.split(','), dtype='float64')
    # 0 column is index -> drop, 1 col label
    return np.array([arr[2:]]), np.array([arr[1]])

#clfs = [SAMKNN(), ARSLVQ(gradient_descent='Adadelta'), ARF()]
clfs = [ARF()]
    
classes = np.arange(0, 15, 1)

transformer = SparseRandomProjection(n_components=1000)

"""TF-IDF"""
# High dim
#for clf in clfs:
#    acc_fold = []
#    kappa_fold = []
#    time_fold = []
#    
#    for _ in range(5):
#        _clf = copy.deepcopy(clf)
#        f = open('data/nasdaq_stream_wo_sentiment.csv')
#        start_time = time.time()
#        y_true = []
#        y_pred = []
#
#        x, y = next_sample()
#        #print(f'{x.shape[1]} dimensions')
#        _clf.partial_fit(x, y, classes=classes)
#        
#        while True:
#            x, y = next_sample()
#            if type(x) == str and x == 'EOF': break
#            # test with less labels
#        #    if y > 9: continue
#            y_pred.append(_clf.predict(x)[0])
#        
#            y_true.append(y[0])
#            # update clf
#            _clf.partial_fit(x, y)
#        
#        f.close()
#        print('high fold done')
#        
#        acc_fold.append(accuracy_score(y_true, y_pred))
#        kappa_fold.append(cohen_kappa_score(y_true, y_pred))
#        time_fold.append(time.time() - start_time)
#        
#    res_file = open('res.txt', 'a+')
#    res_file.write(50 * '-')
#    res_file.write('\n')
#    res_file.write(f'High dim test: \n{_clf}\n')
#    res_file.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
#    res_file.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
#    res_file.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
#    res_file.write(50 * '-')
#    res_file.write('\n\n')
#    res_file.close()
        
# Low Dim
for clf in clfs:
    acc_fold = []
    kappa_fold = []
    time_fold = []
    
    for _ in range(1):
        _clf = copy.deepcopy(clf)
        f = open('data/nasdaq_stream_wo_sentiment.csv')
        start_time = time.time()
        y_true = []
        y_pred = []

        x, y = next_sample()
       # print(f'{x.shape[1]} dimensions')
        # Learn RP
        x = transformer.fit_transform(x)
        _clf.partial_fit(x, y, classes=classes)
        
        while True:
            x, y = next_sample()
            if type(x) == str and x == 'EOF': break
            # test with less labels
        #    if y > 9: continue
            # Perform RP
            x = transformer.transform(x)
            y_pred.append(_clf.predict(x)[0])
        
            y_true.append(y[0])
            # update clf
            _clf.partial_fit(x, y)
        
        f.close()
        print('low fold done')
        
        acc_fold.append(accuracy_score(y_true, y_pred))
        kappa_fold.append(cohen_kappa_score(y_true, y_pred))
        time_fold.append(time.time() - start_time)
        
    res_file = open('res.txt', 'a+')
    res_file.write(50 * '-')
    res_file.write('\n')
    res_file.write(f'Low dim test: \n{_clf}\n')
    res_file.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
    res_file.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
    res_file.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
    res_file.write(50 * '-')
    res_file.write('\n\n')
    res_file.close()
    
"""SKIP-GRAM"""
#f = open('res.txt', 'a+')
#f.write('SKIP-GRAM\n')
#f.close()
#data = np.load('data/skip-gram-embed-w-label.npy')

# HIGH-DIM
#X, y = data[:, :-1], data[:, -1]
#
#for clf in clfs:
#    acc_fold = []
#    kappa_fold = []
#    time_fold = []
#    
#    for _ in range(5):
#        _clf = copy.deepcopy(clf)
#        start_time = time.time()
#        y_true = []
#        y_pred = []
#        
#        x = data[0, :-1].reshape(1, 1000)
#        y = data[0, -1].reshape(1, 1)
#        
#        # pretrain
#        _clf.partial_fit(x, y.ravel(), classes=classes)
#        
#        for i in range(data.shape[0]):
#            x = data[i, :-1].reshape(1, 1000)
#            y = data[i, -1].reshape(1, 1)
#            y_pred.append(_clf.predict(x)[0])
#        
#            y_true.append(y[0])
#            # update clf
#            _clf.partial_fit(x, y.ravel())
#        
#        print('high fold skipgram done')
#        
#        acc_fold.append(accuracy_score(y_true, y_pred))
#        kappa_fold.append(cohen_kappa_score(y_true, y_pred))
#        time_fold.append(time.time() - start_time)
#        
#    res_file = open('res.txt', 'a+')
#    res_file.write(50 * '-')
#    res_file.write('\n')
#    res_file.write(f'High dim test skip-gram: \n{_clf}\n')
#    res_file.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
#    res_file.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
#    res_file.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
#    res_file.write(50 * '-')
#    res_file.write('\n\n')
#    res_file.close()
#
## LOW-DIM
#X, y = data[:, :-1], data[:, -1]
#
#for clf in clfs:
#    acc_fold = []
#    kappa_fold = []
#    time_fold = []
#    
#    for _ in range(5):
#        _clf = copy.deepcopy(clf)
#        start_time = time.time()
#        y_true = []
#        y_pred = []
#        
#        x = data[0, :-1].reshape(1, 1000)
#        y = data[0, -1].reshape(1, 1)
#        
#        # Learn RP
#        x = transformer.fit_transform(x)
#        _clf.partial_fit(x, y, classes=classes)
#        
#        for i in range(data.shape[0]):
#            x = data[i, :-1].reshape(1, 1000)
#            y = data[i, -1].reshape(1, 1)
#            x = transformer.transform(x)
#            y_pred.append(_clf.predict(x)[0])
#        
#            y_true.append(y[0])
#            # update clf
#            _clf.partial_fit(x, y)
#        
#        print('low fold skipgram done')
#        
#        acc_fold.append(accuracy_score(y_true, y_pred))
#        kappa_fold.append(cohen_kappa_score(y_true, y_pred))
#        time_fold.append(time.time() - start_time)
        
#    res_file = open('res.txt', 'a+')
#    res_file.write(50 * '-')
#    res_file.write('\n')
#    res_file.write(f'Low dim test skip-gram: \n{_clf}\n')
#    res_file.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
#    res_file.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
#    res_file.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
#    res_file.write(50 * '-')
#    res_file.write('\n\n')
#    res_file.close()

