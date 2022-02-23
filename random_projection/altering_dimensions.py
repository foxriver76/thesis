#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:54:16 2019

@author: moritz
"""

import time
import copy
from skmultiflow.data import SEAGenerator, LEDGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier as ARF, OzaBaggingADWINClassifier as OBA
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.random_projection import SparseRandomProjection
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # backend which does not show plots

def low_dim_test(stream, clf, n_samples, flip=True):
    """Test in low dimensional space - enrich then project samples"""
    y_true_sum = np.zeros(n_samples - 1)
    y_pred_sum = np.zeros(n_samples - 1)

    stream.next_sample()

    n_rand_dims = 10000 - stream.current_sample_x.size
    multiply = n_rand_dims // stream.current_sample_x.size

    """Iteration for projected dims"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []
    
    for _ in range(5):
        current_sample_x = [[]]
        for _m in range(multiply):
            current_sample_x = np.concatenate(
                    (current_sample_x, stream.current_sample_x), axis=1)

        sparse_transformer_li = SparseRandomProjection(
                n_components=1000, density='auto')

        """Create projection matrix"""
        sparse_transformer_li.fit(current_sample_x)
        
        for i in range(n_samples):
            stream.next_sample()

            """We have to enrich the sample with meaningless random dimensions"""
            # enhance dims
            current_sample_x = [[]]
            for _m in range(multiply):
                current_sample_x = np.concatenate(
                    (current_sample_x, stream.current_sample_x), axis=1)

            if i == 0:
                """Pretrain Classifier"""
                pretrain_size = 500
                stream.next_sample(pretrain_size)
                current_sample_enhanced = [[] for _p in range(pretrain_size)]
                for _m in range(multiply):
                    current_sample_enhanced = np.concatenate(
                        (current_sample_enhanced, stream.current_sample_x), axis=1)
                reduced_x = sparse_transformer_li.transform(
                    current_sample_enhanced)
                clf.partial_fit(
                    reduced_x, stream.current_sample_y.ravel(), classes=stream.target_values)
                start_time = time.time()
                continue
            
            if i / n_samples == 0.25 and flip is True:
                print('reduce matrix')
                old_matrix = sparse_transformer_li.components_.toarray()
                new_matrix = old_matrix[:, 0:9000]
                sparse_transformer_li.components_ = csr_matrix(new_matrix)
            
            if i / n_samples == 0.75 and flip is True:
                print('extend matrix')
                sparse_transformer_li.components_ = csr_matrix(old_matrix)
                
            """from 25 % to 75% remove 1000 dimensions"""
            if i / n_samples >= 0.25 and i / n_samples < 0.75 and flip is True:
                current_sample_x = current_sample_x[:, 0:9000]

            reduced_x = sparse_transformer_li.transform(current_sample_x)

            """Predict then train"""
            y_pred = clf.predict(reduced_x)
            clf.partial_fit(
                reduced_x, stream.current_sample_y.ravel(), classes=stream.target_values)

            """Save true and predicted y"""
            y_true_sum[i - 1] = stream.current_sample_y
            y_pred_sum[i - 1] = y_pred

        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)

        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)
    
    f = open('result_flip.txt', 'a+')
    f.write('NO FLIP\n' if flip is False else 'WITH FLIP\n')
    f.write('Evaluated {} low dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(),
                                                                                        np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} low dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(clf, np.array(time_collect).mean(), np.array(time_collect).std(),
                                                                                      np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    
    """build acc plot"""
    accs = []
    for i in range(N_SAMPLES):
        if i == 0: continue
        accs.append(accuracy_score(y_true_sum[0:i], y_pred_sum[0:i]))
    plt.plot(np.array(accs))
    ax = plt.gca()
    ax.set_ylim([0, 1.05])
    plt.savefig('fig/flip_res_{}_{}.png'.format(stream.name, clf.name))
    plt.clf() # clear plot

if __name__ == '__main__':
    N_SAMPLES = 30000

    """Check accuracy on gradual and abrupt drift streams"""
    """Gradual STAGGER"""
    STREAMS = []

    """SEA"""
    stream = SEAGenerator()
    stream.name = 'SEA'
    STREAMS.append(stream)

    """LED"""
    stream = LEDGenerator()
    stream.name = 'LED'
    STREAMS.append(stream)

    """Evaluate on ARF, OBA"""
    for stream in STREAMS:
        print('{}:\n'.format(stream.name))
        f = open('result_flip.txt', 'a+')
        f.write('{}:\n'.format(stream.name))
        f.close()
        
        arf = ARF()
        arf.name = 'ARF'
        low_dim_test(copy.copy(stream), copy.copy(arf), N_SAMPLES)
        
        arf = ARF()
        arf.name = 'ARF'
        low_dim_test(copy.copy(stream), copy.copy(arf), N_SAMPLES, flip=False)

        oba = OBA()
        oba.name = 'OBA'
        low_dim_test(copy.copy(stream), copy.copy(oba), N_SAMPLES)

        oba = OBA()
        oba.name = 'OBA'
        low_dim_test(copy.copy(stream), copy.copy(oba), N_SAMPLES, flip=False)
        