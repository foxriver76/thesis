#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:03:55 2020

@author: moritz
"""

from skmultiflow.data import SEAGenerator, LEDGeneratorDrift, SineGenerator, MIXEDGenerator
from skmultiflow.drift_detection import ADWIN, EDDM, DDM
from bix.detectors.kswin import KSWIN
from bix.data.reoccuringdriftstream import ReoccuringDriftStream
from skmultiflow.bayes import NaiveBayes
from mebwin import MEBWIN
from sw_mebwin import SWMEBWIN
from kernel_swmebwin import Kernel_SWMEBWIN
import time
import numpy as np
import matplotlib.pyplot as plt
# Interactive mode off
plt.ioff()

stream = ReoccuringDriftStream(stream=SEAGenerator(classification_function=0), drift_stream=SEAGenerator(classification_function=1), width=1, alpha=90, pause=1000, position=2000)

stream.next_sample()

RANGE = 10000
DIM = 50

"""Runtime wrt eps kernel, euclid w_size=100"""
data = []
labels = []

times = {'swmebwin': {}, 'k-swmebwin': {}}


n_rand_dims = DIM - stream.current_sample_x.size
multiply = n_rand_dims // stream.current_sample_x.size
eps_coll = [.5, .3, .1, .01, .001]

for i in range(RANGE):
    current_sample_x = np.array([[]])
    for _m in range(multiply):
        current_sample_x = np.concatenate(
                (current_sample_x, stream.current_sample_x), axis=1)
    data.append(current_sample_x.ravel())
    labels.append(stream.current_sample_y.ravel()[0])
    stream.next_sample()

for eps in eps_coll:
    swmebwin = SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=eps)

    start = time.time()
    for i in range(RANGE):
        swmebwin.add_element(data[i], labels[i])
        
    times['swmebwin'][eps] = time.time() - start
    
for eps in eps_coll:
    kswmebwin = Kernel_SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=eps)

    start = time.time()
    for i in range(RANGE):
        kswmebwin.add_element(data[i], labels[i])
        
    times['k-swmebwin'][eps] = time.time() - start
    
plt.plot(eps_coll, list(times['swmebwin'].values()), color='red', linestyle='dotted', label='SWMEBWIN')
plt.semilogx(eps_coll, list(times['k-swmebwin'].values()), color='blue', label='K-SWMEBWIN')
plt.gca().invert_xaxis()
plt.legend()

plt.xlabel('eps')
plt.ylabel('runtime in seconds')

plt.savefig('../figures/runtime_eps.eps')
plt.clf()

"""Runtime wrt # dims swmebwin k-swmebwin, kswin, adwin, eddm, ddm  10, 25, 50, 100, 500 auf 10k Samples"""
dim_ranges = [10, 25, 50, 100, 500]

time_ksmebwin = []
time_smebwin = []
time_kswin = []
time_adwin = []
time_ddm = []
time_eddm = []
labels = []
predictions = []

for dim_range in dim_ranges:
    data = []
    multiply = dim_range // stream.current_sample_x.size
    
    adwin = []
    kswin = []
    
    for j in range(dim_range):
        adwin.append(ADWIN(delta=0.002))
        kswin.append(KSWIN(w_size=300, stat_size=30, alpha=0.0001))
    
    bayes = NaiveBayes()
    
    # partial fit -> pretrain
    for _m in range(multiply):
        current_sample_x = np.array([[]])
        current_sample_x = np.concatenate(
                    (current_sample_x, stream.current_sample_x), axis=1)
         
    bayes.partial_fit(np.array(current_sample_x), list(stream.current_sample_y.ravel()))
    
    # build data set
    for i in range(RANGE):
        current_sample_x = np.array([[]])
        for _m in range(multiply):
            current_sample_x = np.concatenate(
                    (current_sample_x, stream.current_sample_x), axis=1)
        data.append(current_sample_x.ravel())
        labels.append(stream.current_sample_y.ravel()[0])
        predictions.append(0 if bayes.predict(current_sample_x) == labels[i] else 1)
        bayes.partial_fit(current_sample_x, list(stream.current_sample_y.ravel()))
        stream.next_sample()
        
    # Kernel SWMEBWIN
    kswmebwin = Kernel_SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=0.1)
    start = time.time()
    
    for i in range(RANGE):
        kswmebwin.add_element(data[i], labels[i])
    
    time_ksmebwin.append(time.time() - start)
    
    # SWMEBWIN
    swmebwin = SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=0.1)
    start = time.time()
    
    for i in range(RANGE):
        swmebwin.add_element(data[i], labels[i])
    
    time_smebwin.append(time.time() - start)
    
    # KSWIN
    start = time.time()
    
    for i in range(RANGE):            
        for j in range(data[i].size):    
            kswin[j].add_element(data[i][j])

    time_kswin.append(time.time() - start)
    
    # ADWIN
    start = time.time()
    
    for i in range(RANGE):            
        for j in range(data[i].size):    
            adwin[j].add_element(data[i][j])

    time_adwin.append(time.time() - start)
    
    # DDM
    ddm = DDM(min_num_instances=30)
    start = time.time()
    
    for i in range(RANGE):            
        ddm.add_element(predictions[i])

    time_ddm.append(time.time() - start)
    
    # EDDM
    eddm = EDDM()
    start = time.time()
    
    for i in range(RANGE):            
        eddm.add_element(predictions[i])

    time_eddm.append(time.time() - start)

# now visualize the drifts
plt.plot(dim_ranges, time_smebwin, color='red', label='MEBWIND')
plt.plot(dim_ranges, time_ksmebwin, color='blue', label='K-MEBWIND', linestyle='dotted', marker='o')
plt.plot(dim_ranges, time_kswin, color='orange', label='KSWIN', linestyle='dashdot', marker=4)
plt.plot(dim_ranges, time_adwin, color='green', label='ADWIN', linestyle='dashed', marker='D')
plt.plot(dim_ranges, time_eddm, color='purple', label='EDDM', linestyle=(0, (1, 10)), marker='+')
plt.plot(dim_ranges, time_ddm, color='black', label='DDM', linestyle=(0, (3, 5, 1, 5, 1, 5)), marker='*')

plt.legend(loc='best', bbox_to_anchor=(0.99, 0.6))

plt.yscale('log')
plt.xscale('log')

plt.xlabel('# dimensions')
plt.ylabel('runtime in seconds')

plt.savefig('../figures/runtime_dims_log_log.eps')

plt.clf()

"""Runtime wrt # samples swmebwin k-swmebwin, kswin, adwin, eddm, ddm 100, 1000, 10000, 100000"""
predictions = []
data = []
labels= []

bayes = NaiveBayes()

n_rand_dims = DIM - stream.current_sample_x.size
multiply = n_rand_dims // stream.current_sample_x.size

# partial fit -> pretrain
for _m in range(multiply):
    current_sample_x = np.array([[]])
    current_sample_x = np.concatenate(
                (current_sample_x, stream.current_sample_x), axis=1)
     
bayes.partial_fit(np.array(current_sample_x), list(stream.current_sample_y.ravel()))

sample_ranges = [100, 1000, 10000, 100000, 1000000]

for i in range(sample_ranges[-1]):
    current_sample_x = np.array([[]])
    for _m in range(multiply):
        current_sample_x = np.concatenate(
                (current_sample_x, stream.current_sample_x), axis=1)
    data.append(current_sample_x.ravel())
    labels.append(stream.current_sample_y.ravel()[0])
    predictions.append(0 if bayes.predict(current_sample_x) == labels[i] else 1)
    bayes.partial_fit(current_sample_x, list(stream.current_sample_y.ravel()))
    stream.next_sample()

print('Dataset created')

time_ksmebwin = []
time_smebwin = []
time_kswin = []
time_adwin = []
time_ddm = []
time_eddm = []

for sample_range in sample_ranges:
    print(f'now working on sample range {sample_range}')
    # Initialize detectors
    adwin = []
    kswin = []
    
    for j in range(DIM):
        adwin.append(ADWIN(delta=0.002))
        kswin.append(KSWIN(w_size=300, stat_size=30, alpha=0.0001))
    
    kswmebwin = Kernel_SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=0.1)
    swmebwin = SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=0.1)

    # Kernel SWMEBWIN
    start = time.time()
    
    for i in range(sample_range):
        kswmebwin.add_element(data[i], labels[i])

    time_ksmebwin.append(time.time() - start)
    
    # SWMEBWIN
    swmebwin = SWMEBWIN(classes=stream.target_values, w_size=100, epsilon=0.1)
    start = time.time()
    
    for i in range(sample_range):
        swmebwin.add_element(data[i], labels[i])
    
    time_smebwin.append(time.time() - start)
    
    # KSWIN
    start = time.time()
    
    for i in range(sample_range):            
        for j in range(data[i].size):    
            kswin[j].add_element(data[i][j])

    time_kswin.append(time.time() - start)
    
    # ADWIN
    start = time.time()
    
    for i in range(sample_range):            
        for j in range(data[i].size):    
            adwin[j].add_element(data[i][j])

    time_adwin.append(time.time() - start)
    
    # DDM
    ddm = DDM(min_num_instances=30)
    start = time.time()
    
    for i in range(RANGE):            
       ddm.add_element(predictions[i])

    time_ddm.append(time.time() - start)
    
    # EDDM
    eddm = EDDM()
    start = time.time()
    
    for i in range(RANGE):            
       eddm.add_element(predictions[i])

    time_eddm.append(time.time() - start)
        
# now visualize the runtime
plt.plot(sample_ranges, time_smebwin, color='red', label='MEBWIND')
plt.plot(sample_ranges, time_ksmebwin, color='blue', label='K-MEBWIND', linestyle='dotted', marker='o')
plt.plot(sample_ranges, time_kswin, color='orange', label='KSWIN', linestyle='dashdot', marker=4)
plt.plot(sample_ranges, time_adwin, color='green', label='ADWIN', linestyle='dashed', marker='D')
plt.plot(sample_ranges, time_eddm, color='purple', label='EDDM', linestyle=(0, (1, 10)), marker='+')
plt.plot(sample_ranges, time_ddm, color='black', label='DDM', linestyle=(0, (3, 5, 1, 5, 1, 5)), marker='*')

plt.legend()
plt.yscale('log')
plt.xscale('log')

plt.xlabel('# samples')
plt.ylabel('runtime in seconds')

plt.savefig('../figures/runtime_samples_log_log.eps')

plt.clf()