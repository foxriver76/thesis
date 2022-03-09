#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 12:46:07 2020

@author: moritz
"""

import sys, os
sys.path.append(os.path.abspath(__file__ + "/../../"))
from skmultiflow.data import SEAGenerator, LEDGeneratorDrift, SineGenerator, MIXEDGenerator
from skmultiflow.drift_detection import ADWIN, EDDM, DDM, KSWIN
from prototype_lvq.utils.reoccuring_drift_stream import ReoccuringDriftStream
#from mebwin import MEBWIN
from lib.sw_mebwin import SWMEBWIN
from lib.kernel_swmebwin import Kernel_SWMEBWIN
import time
import numpy as np
import typing
from skmultiflow.bayes import NaiveBayes


streams = []

"""CD at 2000 and then every 1000"""
# MIXED
stream = ReoccuringDriftStream(stream=MIXEDGenerator(classification_function=0), drift_stream=MIXEDGenerator(classification_function=1), width=1, alpha=90, pause=1000, position=2000)

stream.name = 'MIXED Abrupt Reoccuring'
streams.append(stream)

# SEA
stream = ReoccuringDriftStream(stream=SEAGenerator(classification_function=0), drift_stream=SEAGenerator(classification_function=1), width=1, alpha=90, pause=1000, position=2000)

stream.name = 'SEA Abrupt Reoccuring'
streams.append(stream)

# SINE
stream = ReoccuringDriftStream(stream=SineGenerator(classification_function=0), drift_stream=SineGenerator(classification_function=1), width=1, alpha=90, pause=1000, position=2000)

stream.name = 'Sine Abrupt Reoccuring'
streams.append(stream)

# LED
stream = ReoccuringDriftStream(stream=LEDGeneratorDrift(noise_percentage=0.2, has_noise=True, n_drift_features=5), drift_stream=LEDGeneratorDrift(noise_percentage=0, has_noise=True, n_drift_features=10), width=1, alpha=90, pause=1000, position=2000)

stream.name = 'LED Abrupt Reoccuring'
streams.append(stream)

# TODO: lets calc conf matrix over all streams

def main():
    
    overall_kswin_tp = overall_kswin_tn = overall_kswin_fp = overall_kswin_fn = 0
    overall_adwin_tp = overall_adwin_tn = overall_adwin_fp = overall_adwin_fn = 0
#   mebwin_drifts = []
    overall_k_swmebwin_tp = overall_k_swmebwin_tn = overall_k_swmebwin_fp = overall_k_swmebwin_fn = 0
    overall_swmebwin_tp = overall_swmebwin_tn = overall_swmebwin_fp = overall_swmebwin_fn = 0
    overall_eddm_tp = overall_eddm_tn = overall_eddm_fp = overall_eddm_fn = 0
    overall_ddm_tp = overall_ddm_tn = overall_ddm_fp = overall_ddm_fn = 0
    
    for stream in streams:
        print(stream.name)
        
        f = open('drifts.txt', 'a+')
        f.write(f'**{stream.name}**\n\n')
        f.close()
                
        stream.prepare_for_use()
        
        stream.next_sample()
        
#        mebwin = MEBWIN(epsilon=0.1, sensitivity=0.98, w_size=100, stat_size=30)
        adwin = []
        kswin = []
        ddm = DDM(min_num_instances=30)
        eddm = EDDM()
        
        data = []
        labels = []
        predictions = []
        
        kswin_drifts = []
        adwin_drifts = []
#        mebwin_drifts = []
        k_swmebwin_drifts = []
        swmebwin_drifts = []
        eddm_drifts = []
        ddm_drifts = []
        
        swmebwin = SWMEBWIN(classes=stream.target_values, w_size=80, epsilon=0.05)
#        k_swmebwin = Kernel_SWMEBWIN(classes=stream.target_values, w_size=80, epsilon=0.05, gamma=10**10)
        k_swmebwin = Kernel_SWMEBWIN(classes=stream.target_values, w_size=80, epsilon=0.05)
        # gamma maybe 1.0 / stream.current_sample_x.shape[1]
        RANGE = 1000000
        DIM = 50
        # - 2 because first drift is at 2000 not 1000 and last drift is not detectable
#        COUNT_DRIFTS = RANGE / 1000 - 2
        
        n_rand_dims = DIM - stream.current_sample_x.size
        multiply = n_rand_dims // stream.current_sample_x.size
        
        # partial fit -> pretrain
        for _m in range(multiply):
            current_sample_x = np.array([[]])
            current_sample_x = np.concatenate(
                        (current_sample_x, stream.current_sample_x), axis=1)
     
        bayes = NaiveBayes()
        bayes.partial_fit(np.array(current_sample_x), list(stream.current_sample_y.ravel()))
        
        for j in range(DIM):
            adwin.append(ADWIN(delta=0.002))
            kswin.append(KSWIN(window_size=300, stat_size=30, alpha=0.0001))
                    
        """Add dims"""
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
        
        # MEBWIN
    #    start = time.time()
    #    for i in range(RANGE):
    #        mebwin.add_element(data[i])
    #        
    #        if mebwin.change_detected is True:
    #            mebwin_drifts.append(i)
    #
    #    f = open('drifts.txt', 'a+')
    #    f.write(f'MEBWIN detected {len(mebwin_drifts)} drifts in {time.time() - start} {mebwin_drifts}\n\n')
    #    f.close() 
    #    print(f'MEBWIN took {time.time() - start} sec and detected {len(mebwin_drifts)} drifts')
    
        # Kernel SWMEBWIN
        start = time.time()
        for i in range(RANGE):
            k_swmebwin.add_element(value=data[i], label=labels[i])
            
            if k_swmebwin.change_detected is True:
                k_swmebwin_drifts.append(i)
          
        end = time.time() - start
    
        f1, tp, fp, tn, fn = confusion_matrix_stats(k_swmebwin_drifts, RANGE)
        overall_k_swmebwin_tp += tp
        overall_k_swmebwin_tn += tn
        overall_k_swmebwin_fp += fp
        overall_k_swmebwin_fn += fn
    
        print(f'F1-Score: {f1}')
        print(f'{tp} true positives, {fp} false positives')
        print(f'{tn} true negatives, {fn} false negatives')
            
        f = open('drifts.txt', 'a+')
        f.write(f'K-SWMEB detected {len(k_swmebwin_drifts)} drifts in {time.time() - start} {k_swmebwin_drifts}\n\n')
        f.close()
        print(f'K-SW-MEBWIN took {end} sec and detected {len(k_swmebwin_drifts)} drifts\n')
             
        # SWMEBWIN
        start = time.time()
        for i in range(RANGE):
            swmebwin.add_element(value=data[i], label=labels[i])
            
            if swmebwin.change_detected is True:
                swmebwin_drifts.append(i)
          
        end = time.time() - start
    
        f1, tp, fp, tn, fn = confusion_matrix_stats(swmebwin_drifts, RANGE)
        
        overall_swmebwin_tp += tp
        overall_swmebwin_tn += tn
        overall_swmebwin_fp += fp
        overall_swmebwin_fn += fn
    
        print(f'F1-Score: {f1}')
        print(f'{tp} true positives, {fp} false positives')
        print(f'{tn} true negatives, {fn} false negatives')
            
        f = open('drifts.txt', 'a+')
        f.write(f'SWMEB detected {len(swmebwin_drifts)} drifts in {time.time() - start} {swmebwin_drifts}\n\n')
        f.close()
        print(f'SW-MEBWIN took {end} sec and detected {len(swmebwin_drifts)} drifts\n')
                
        # ADWIN
        start = time.time()
        for i in range(RANGE):
            adwin_detected = False
        
            for j in range(data[i].size):
                adwin[j].add_element(data[i][j])
                if adwin[j].detected_change():
                    adwin_detected = True
                    
            if adwin_detected is True:
                adwin_drifts.append(i)
                
        end = time.time() - start
        
        f1, tp, fp, tn, fn = confusion_matrix_stats(adwin_drifts, RANGE)
        
        overall_adwin_tp += tp
        overall_adwin_tn += tn
        overall_adwin_fp += fp
        overall_adwin_fn += fn
    
        print(f'F1-Score: {f1}')
        print(f'{tp} true positives, {fp} false positives')
        print(f'{tn} true negatives, {fn} false negatives')
            
        f = open('drifts.txt', 'a+')
        f.write(f'ADWIN detected {len(adwin_drifts)} drifts in {time.time() - start} at {adwin_drifts}\n\n')
        f.close()
        print(f'ADWIN took {end} sec and detected {len(adwin_drifts)} drifts\n')
        
        # KSWIN
        start = time.time()
        for i in range(RANGE):
            kswin_detected = False
            
            for j in range(data[i].size):    
                kswin[j].add_element(data[i][j])
                if kswin[j].detected_change():
                    kswin_detected = True
                    
            if kswin_detected is True:
                kswin_drifts.append(i)
          
        end = time.time() - start
        
        f1, tp, fp, tn, fn = confusion_matrix_stats(kswin_drifts, RANGE)
        
        overall_kswin_tp += tp
        overall_kswin_tn += tn
        overall_kswin_fp += fp
        overall_kswin_fn += fn
    
        print(f'F1-Score: {f1}')
        print(f'{tp} true positives, {fp} false positives')
        print(f'{tn} true negatives, {fn} false negatives')     
        
        f = open('drifts.txt', 'a+')
        f.write(f'KSWIN detected {len(kswin_drifts)} drifts in {time.time() - start} at {kswin_drifts}\n\n')
        f.close()
        print(f'KSWIN took {end} sec and detected {len(kswin_drifts)} drifts\n')
        
        # EDDM
        start = time.time()
        for i in range(RANGE):
            eddm_detected = False
            
            eddm.add_element(predictions[i])
            
            if eddm.detected_change():
                eddm_detected = True
                    
            if eddm_detected is True:
                eddm_drifts.append(i)
                
        end = time.time() - start
          
        f1, tp, fp, tn, fn = confusion_matrix_stats(eddm_drifts, RANGE)
        
        overall_eddm_tp += tp
        overall_eddm_tn += tn
        overall_eddm_fp += fp
        overall_eddm_fn += fn
    
        print(f'F1-Score: {f1}')
        print(f'{tp} true positives, {fp} false positives')
        print(f'{tn} true negatives, {fn} false negatives')
        
        f = open('drifts.txt', 'a+')
        f.write(f'EDDM detected {len(eddm_drifts)} drifts in {time.time() - start} at {eddm_drifts}\n\n')
        f.close()
        print(f'EDDM took {end} sec and detected {len(eddm_drifts)} drifts\n')
        
        # DDM
        start = time.time()
        for i in range(RANGE):
            ddm_detected = False
            ddm.add_element(predictions[i])
            if ddm.detected_change():
                ddm_detected = True
                    
            if ddm_detected is True:
                ddm_drifts.append(i)
                
        end = time.time() - start
        
        f1, tp, fp, tn, fn = confusion_matrix_stats(ddm_drifts, RANGE)
        
        overall_ddm_tp += tp
        overall_ddm_tn += tn
        overall_ddm_fp += fp
        overall_ddm_fn += tn
    
        print(f'F1-Score: {f1}')
        print(f'{tp} true positives, {fp} false positives')
        print(f'{tn} true negatives, {fn} false negatives')
        
        f = open('drifts.txt', 'a+')
        f.write(f'DDM detected {len(ddm_drifts)} drifts in {time.time() - start} at {ddm_drifts}\n\n')
        f.close()
        print(f'DDM took {end} sec and detected {len(ddm_drifts)} drifts\n')
        
    # OVERALL STATISTICS
    print(50 * '-')
    print('K-SWMEBWIN\n')
    print(f'Overall F1: {calc_f1(overall_k_swmebwin_tp, overall_k_swmebwin_fp, overall_k_swmebwin_tn, overall_k_swmebwin_fn)}')
    print(f'{overall_k_swmebwin_tp} true positives, {overall_k_swmebwin_fp} false positives')
    print(f'{overall_k_swmebwin_tn} true negatives, {overall_k_swmebwin_fn} false negatives')
    print(50* '-')
    
    print(50 * '-')
    print('SWMEBWIN\n')
    print(f'Overall F1: {calc_f1(overall_swmebwin_tp, overall_swmebwin_fp, overall_swmebwin_tn, overall_swmebwin_fn)}')
    print(f'{overall_swmebwin_tp} true positives, {overall_swmebwin_fp} false positives')
    print(f'{overall_swmebwin_tn} true negatives, {overall_swmebwin_fn} false negatives')
    print(50* '-')
    
    print(50 * '-')
    print('KSWIN\n')
    print(f'Overall F1: {calc_f1(overall_kswin_tp, overall_kswin_fp, overall_kswin_tn, overall_kswin_fn)}')
    print(f'{overall_kswin_tp} true positives, {overall_kswin_fp} false positives')
    print(f'{overall_kswin_tn} true negatives, {overall_kswin_fn} false negatives')
    print(50* '-')
    
    print(50 * '-')
    print('ADWIN\n')
    print(f'Overall F1: {calc_f1(overall_adwin_tp, overall_adwin_fp, overall_adwin_tn, overall_adwin_fn)}')
    print(f'{overall_adwin_tp} true positives, {overall_adwin_fp} false positives')
    print(f'{overall_adwin_tn} true negatives, {overall_adwin_fn} false negatives')
    print(50* '-')
    
    print(50 * '-')
    print('DDM\n')
    print(f'Overall F1: {calc_f1(overall_ddm_tp, overall_ddm_fp, overall_ddm_tn, overall_ddm_fn)}')
    print(f'{overall_ddm_tp} true positives, {overall_ddm_fp} false positives')
    print(f'{overall_ddm_tn} true negatives, {overall_ddm_fn} false negatives')
    print(50* '-')
    
    print(50 * '-')
    print('EDDM\n')
    print(f'Overall F1: {calc_f1(overall_eddm_tp, overall_eddm_fp, overall_eddm_tn, overall_eddm_fn)}')
    print(f'{overall_eddm_tp} true positives, {overall_eddm_fp} false positives')
    print(f'{overall_eddm_tn} true negatives, {overall_eddm_fn} false negatives')
    print(50* '-')
    
def calc_f1(tp, fp, tn, fn):
    try:
        precision = tp / (tp + fp)
    except Exception as e:
        print(f'Could not calc f1 - precision: {e}')
        precision = 0
    
    try:
        recall = tp / (tp + fn)
    except Exception as e:
        print(f'Could not calc f1 - recall: {e}')
        recall = 0
        
    try:
        return 2 * (precision * recall / (precision + recall))
    except:
        return 0

def confusion_matrix_stats(drifts:[], RANGE:int) -> typing.Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0 
    drift_active = False
    for i in range(RANGE):
        if (i % 1000 <= 50 or i % 1000 == 999) and drift_active is False and i > 1900:
            # in this window it would be true positive
            try:
                drifts.index(i)
                drift_active = True
                tp += 1
            except ValueError:
                tn += 1
        elif i % 1000 == 51 and drift_active is False and i > 1900:
            # drift has not been detected in window -> window over
            fn += 1     
        else:
            # no drift
            if not (i % 1000 <= 50 or i % 1000 == 999):
                drift_active = False
            try:
                drifts.index(i)
                # drift detected but there is no
                fp += 1
            except ValueError:
                # no drift detected -> correct
                tn += 1
    f1 = calc_f1(tp, fp, tn, fn)
    return f1, tp, fp, tn, fn

if __name__ == '__main__':
    main()