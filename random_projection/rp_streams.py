#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:31:21 2019

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from skmultiflow.data import ConceptDriftStream, STAGGERGenerator, SEAGenerator, FileStream, LEDGeneratorDrift, MIXEDGenerator, SineGenerator
from skmultiflow.lazy import SAMKNN
from skmultiflow.meta import AdaptiveRandomForest as ARF
from skmultiflow.trees import HAT
from sklearn.random_projection import SparseRandomProjection
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import time
from rff_base import Base as RFF
from random_projection.model.rrslvq import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ
from rslvq import RSLVQ

import copy
res_file="rp_result_stream1_1.txt"
def low_dim_test(stream, clf, n_samples):
    y_true_sum = np.zeros(n_samples - 1)
    y_pred_sum = np.zeros(n_samples - 1)
    
    stream.prepare_for_use()
    stream.next_sample()
    
    n_rand_dims = 10000 - stream.current_sample_x.size
    current_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)
    
    sparse_transformer_li = SparseRandomProjection(n_components=1000, density='auto')
    
    """Create projection matrix"""
    sparse_transformer_li.fit(current_sample_x)

    """Iteration for projected dims"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []
    for j in range(3):
        print(str(j))
        start_time = time.time()
        for i in range(n_samples):
            stream.next_sample()
            
            """We have to enrich the sample with meaningless random dimensions"""
            current_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)
            
            if i == 0:
                """Pretrain Classifier"""
                PRETRAIN_SIZE = 500
                stream.next_sample(PRETRAIN_SIZE)
                current_sample_enhanced = np.array([np.append(stream.current_sample_x[i], np.random.randint(2, size=n_rand_dims)) for i in range(PRETRAIN_SIZE)])
                reduced_x = sparse_transformer_li.transform(current_sample_enhanced)
                clf.partial_fit(reduced_x, stream.current_sample_y.ravel(), classes=stream.target_values)
                continue
            
            reduced_x = sparse_transformer_li.transform(current_sample_x)
    
            """Predict then train"""
            y_pred = clf.predict(reduced_x)
            clf.partial_fit(reduced_x, stream.current_sample_y.ravel(), classes=stream.target_values)
            
            """Save true and predicted y"""
            y_true_sum[i-1] = stream.current_sample_y
            y_pred_sum[i-1] = y_pred
            
        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)
        
        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)
    
    f = open(res_file, 'a+')
    f.write('Evaluated {} {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(stream,clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(stream,clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))


def high_dim_test(stream, clf, n_samples):
    y_true_sum = np.zeros(n_samples - 1)
    y_pred_sum = np.zeros(n_samples - 1)
    
    stream.prepare_for_use()
    stream.next_sample()
    
    n_rand_dims = 10000 - stream.current_sample_x.size
    
    """Iteration for original dim"""
    """5 fold CV"""
    kappa_collect = []
    acc_collect = []
    time_collect = []
    
    for j in range(3):
        print(str(j))
        start_time = time.time()
        for i in range(n_samples):
            stream.next_sample()
            
            """We have to enrich the sample with meaningless random dimensions"""
            current_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(1, stream.n_features + n_rand_dims)
            if i == 0:
                """Pretrain Classifier"""
                PRETRAIN_SIZE = 500
                stream.next_sample(PRETRAIN_SIZE)
                current_sample_enhanced = np.array([np.append(stream.current_sample_x[i], np.random.randint(2, size=n_rand_dims)) for i in range(PRETRAIN_SIZE)])
#                current_sample_x = np.append(stream.current_sample_x, np.random.randint(2, size=n_rand_dims)).reshape(200, stream.n_features + n_rand_dims)
                clf.partial_fit(current_sample_enhanced, stream.current_sample_y.ravel(), classes=stream.target_values)
                continue
                
            """Predict then train"""
            y_pred = clf.predict(current_sample_x)
            clf.partial_fit(current_sample_x, stream.current_sample_y.ravel(), classes=stream.target_values)
            
            """Save true and predicted y"""
            y_true_sum[i-1] = stream.current_sample_y
            y_pred_sum[i-1] = y_pred
            
        """When finished calc acc score"""
        time_sum = time.time() - start_time
        acc = accuracy_score(y_true_sum, y_pred_sum)
        kappa = cohen_kappa_score(y_true_sum, y_pred_sum)
        
        time_collect.append(time_sum)
        kappa_collect.append(kappa)
        acc_collect.append(acc)
    
    f = open(res_file, 'a+')
    f.write('Evaluated {} {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(stream,clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    f.close()
    print('Evaluated {} {} high dim\nTime: {}+-{}\nAcc: {}+-{}\nKappa: {}+-{}\n\n'.format(stream,clf, np.array(time_collect).mean(), np.array(time_collect).std(), np.array(acc_collect).mean(), np.array(acc_collect).std(), np.array(kappa_collect).mean(), np.array(kappa_collect).std()))
    
if __name__ == '__main__':

    N_SAMPLES = 50000
    n_samples = 50000
    """Check accuracy on gradual and abrupt drift streams"""
    """Gradual STAGGER"""
    STREAMS = []

    """Gradual SEA"""
    stream = ConceptDriftStream(SEAGenerator(classification_function=0),
                                SEAGenerator(classification_function=2),
                                position=N_SAMPLES/2,
                                width=N_SAMPLES/5)
    stream.name = 'SEA GRADUAL'
    STREAMS.append(stream)

    """Abrupt SEA"""
    stream = ConceptDriftStream(SEAGenerator(classification_function=0),
                                SEAGenerator(classification_function=1),
                                alpha=90.0, position=N_SAMPLES / 2)
    stream.name = 'SEA ABRUPBT'
    STREAMS.append(stream)

    """ABRUPT LED"""
    stream = ConceptDriftStream(LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                drift_stream=LEDGeneratorDrift(
                                    has_noise=False, noise_percentage=0.0, n_drift_features=7),
                                alpha=90.0, position=N_SAMPLES / 2)
    stream.name = 'LED ABRUPBT'
    STREAMS.append(stream)
    """GRADUAL LED"""
    stream = ConceptDriftStream(LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                                drift_stream=LEDGeneratorDrift(
                                    has_noise=False, noise_percentage=0.0, n_drift_features=7),
                                width=N_SAMPLES / 5, position=N_SAMPLES / 2)
    stream.name = 'LED GRADUAL'
    STREAMS.append(stream)

    for stream in STREAMS:
        print('{}:\n'.format(stream.name))
        """Evaluate on ARSLVQ, SAM and HAT"""
        rrslvq = RRSLVQ(prototypes_per_class=2,confidence=1e-10)
        high_dim_test(copy.copy(stream), copy.copy(rrslvq), n_samples)
        low_dim_test(copy.copy(stream), copy.copy(rrslvq), n_samples)

        arslvq = RSLVQ(gradient_descent='Adadelta')
        high_dim_test(copy.copy(stream), copy.copy(arslvq), N_SAMPLES)
        low_dim_test(copy.copy(stream), copy.copy(arslvq), N_SAMPLES)

        samknn = SAMKNN(max_window_size=5000,stm_size_option=None)
        high_dim_test(copy.copy(stream), copy.copy(samknn), N_SAMPLES)
        low_dim_test(copy.copy(stream), copy.copy(samknn), N_SAMPLES)

        arf = ARF()
        high_dim_test(copy.copy(stream), copy.copy(arf), N_SAMPLES)
        low_dim_test(copy.copy(stream), copy.copy(arf), N_SAMPLES)