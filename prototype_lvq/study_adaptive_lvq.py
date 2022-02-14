#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:50:24 2019

@author: moritz
"""

import pathlib
this_dir = pathlib.Path(__file__).parent.resolve()
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data import SEAGenerator, ConceptDriftStream, LEDGeneratorDrift, HyperplaneGenerator, RandomRBFGeneratorDrift, SineGenerator, FileStream
import time
import numpy as np
import copy
import itertools

# disable the stream generator warnings
import warnings
warnings.filterwarnings('ignore')

def grid_search(clf, stream):
    matrix = list(itertools.product(*[list(v) for v in clf.search_grid.values()]))
    stream.prepare_for_use()
    best_acc = 0
    best_model = copy.deepcopy(clf)
    for param_tuple in matrix:
        try: 
            clf.reset()
        except NotImplementedError: 
            clf.__init__()
            
        for i, param in enumerate(param_tuple):
            clf.__dict__[list(clf.search_grid.keys())[i]] = param
        # now evaluate this classifier
        evaluator = EvaluatePrequential(max_samples=30000)
        evaluator.evaluate(copy.deepcopy(stream), copy.deepcopy(clf))
        curr_acc = evaluator.get_measurements()[0][0].accuracy_score()
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_model = copy.deepcopy(clf)
#            f = open('gs.txt', 'a+')
#            f.write(f'New best model with acc {best_acc} is {best_model}\n')
#            f.close()
    return best_model

def five_fold(stream, clf):
    acc_fold = []
    kappa_fold = []
    runtime_fold = []
    _stream = copy.deepcopy(stream)
    for _ in range(5):
        _stream.prepare_for_use()
        evaluator = EvaluatePrequential(max_samples=1000000)
        start = time.time()
        evaluator.evaluate(_stream, copy.deepcopy(clf))
        rt = time.time() - start
        
        runtime_fold.append(rt)
        acc_fold.append(evaluator.mean_eval_measurements[0].accuracy_score())
        kappa_fold.append(evaluator.mean_eval_measurements[0].kappa_score())
    
    f = open('res.txt', 'a+')
    f.write('{} on {}:\n'.format(clf.name, stream.name))
    f.write('Acc: {} +- {}\n'.format(np.array(acc_fold).mean(), np.array(acc_fold).std()))
    f.write('Kappa: {} +- {}\n'.format(np.array(kappa_fold).mean(), np.array(kappa_fold).std()))
    f.write('Runtime: {} +- {}\n\n'.format(np.array(runtime_fold).mean(), np.array(runtime_fold).std()))
    f.close()
    
#    f = open('res_tex_acc.txt', 'a+')
#    f.write(' & ${} \\pm {}$'.format(round(np.array(acc_fold).mean() * 100, 2), round(np.array(acc_fold).std() * 100, 2)))
#    f.close()
#    
#    f = open('res_tex_kappa.txt', 'a+')
#    f.write(' & ${} \\pm {}$'.format(round(np.array(kappa_fold).mean() * 100, 2), round(np.array(kappa_fold).std() * 100, 2)))
#    f.close()
#    
#    f = open('res_tex_time.txt', 'a+')
#    f.write(' & ${} \\pm {}$'.format(round(np.array(runtime_fold).mean(), 2), round(np.array(runtime_fold).std(), 2)))
#    f.close()
        
if __name__ == '__main__':
    from model.aglvq import GLVQ
    from model.arslvq import RSLVQ
    from skmultiflow.trees import HoeffdingTreeClassifier as HoeffdingTree, HoeffdingAdaptiveTreeClassifier as HAT
    from skmultiflow.meta import OzaBaggingClassifier as OzaBagging, OzaBaggingADWINClassifier as OzaBaggingAdwin
    from skmultiflow.lazy import SAMKNNClassifier as SAMKNN
    
    """Create classifiers"""
    clfs = []
    streams = []
    
#    clf = GLVQ(gradient_descent='SGD')
#    clf.name = 'GLVQ SGD'
#    clf.search_grid = None
#    clfs.append(clf)
    
    clf = GLVQ(gradient_descent='Adadelta')
    clf.name = 'GLVQ Adadelta'
    clf.search_grid = {
            'decay_rate': [0.9, 0.8, 0.9999, 1.0],
            'prototypes_per_class': [1, 2, 4],
            'beta': [1, 2],
            'gradient_descent': ['Adadelta']
            }
    clfs.append(clf)
    
    clf = GLVQ(gradient_descent='Adamax')
    clf.name = 'GLVQ Adamax'
    clf.search_grid = {
    'learning_rate': [0.0001, 0.01, 1e-8, 0.00001, 0.1],
    'beta1': [0.999, 0.9, 0.7, 0.9999999],
    'beta2': [0.999, 0.9, 0.7, 0.9999999],
    'prototypes_per_class': [1, 2, 4],
    'beta': [1, 2],
    'gradient_descent': ['Adamax']
        }
    clfs.append(clf)
    
    clf = RSLVQ(gradient_descent='SGD')
    clf.name = 'RSLVQ SGD'
    clf.search_grid = None
    clfs.append(clf)
    
    clf = RSLVQ(gradient_descent='Adadelta')
    clf.name = 'RSLVQ Adadelta'
    clf.search_grid = {
        'sigma': [0.5, 1, 2, 5],
        'decay_rate': [0.9, 0.8, 0.9999, 1.0],
        'prototypes_per_class': [1, 2, 4],
        'gradient_descent': ['Adadelta']
        }
    clfs.append(clf)
    
    clf = RSLVQ(gradient_descent='Adamax')
    clf.name = 'RSLVQ Adamax'
    clf.search_grid = {
        'sigma': [0.5, 1, 2, 5],
        'learning_rate': [0.0001, 0.01, 1e-8, 0.00001, 0.1],
        'beta1': [0.999, 0.9, 0.7, 0.9999999],
        'beta2': [0.999, 0.9, 0.7, 0.9999999],
        'prototypes_per_class': [1, 2, 4],
        'gradient_descent': ['Adamax']
            }
    clfs.append(clf)
    
    clf = HoeffdingTree()
    clf.name = 'HT'
    clf.search_grid = None
    clfs.append(clf)
    
    clf = HAT()
    clf.name = 'HAT'
    clf.search_grid = None
    clfs.append(clf)
    
    clf = OzaBagging()
    clf.name = 'OB'
    clf.search_grid = None   
    clfs.append(clf)
    
    clf = OzaBaggingAdwin()
    clf.name = 'OBA'
    clf.search_grid = None    
    clfs.append(clf)
    
    clf = SAMKNN()
    clf.name ='SAMKNN'
    clf.search_grid = None    
    clfs.append(clf)
    
    """Create streams"""
    stream = stream = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=2, noise_percentage=0.1),
                            alpha=90.0,
                            random_state=None,
                            position=250000,
                            width=1)
    stream.name = 'SEA$_A$'
    streams.append(stream)
    
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=112, noise_percentage=0.1), 
                            drift_stream=SEAGenerator(random_state=112, 
                                                          classification_function=1, noise_percentage=0.1),
                            random_state=None,
                            position=250000,
                            width=50000)
    stream.name = 'SEA$_G$'
    streams.append(stream)
    
    stream = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            alpha=90.0, # angle of change degree 0 - 90
                            position=250000,
                            width=1)
    stream.name = 'LED$_A$'
    streams.append(stream)
    
    stream = ConceptDriftStream(stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=3),
                            drift_stream=LEDGeneratorDrift(has_noise=False, noise_percentage=0.0, n_drift_features=7),
                            random_state=None,
                            position=250000,
                            width=50000)
    stream.name = 'LED$_G$'
    streams.append(stream)
    
    stream = ConceptDriftStream(stream=SineGenerator(random_state=112, classification_function=0),
                            drift_stream=SineGenerator(random_state=112, classification_function=1),
                            random_state=None,
                            alpha=90.0, # angle of change degree 0 - 90
                            position=250000,
                            width=1)
    stream.name = 'Sine$_A$'
    streams.append(stream)
    
    stream = ConceptDriftStream(stream=SineGenerator(random_state=112, classification_function=0),
                            drift_stream=SineGenerator(random_state=112, classification_function=1),
                            random_state=None,
                            position=250000,
                            width=50000)
    stream.name = 'Sine$_A$'
    streams.append(stream)
    
    stream = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1)
    stream.name = 'HYPERPLANE'
    streams.append(stream)
    
    stream = RandomRBFGeneratorDrift(change_speed=0.001)
    stream.name = 'RBF$_{IF}$'
    streams.append(stream)
    
    stream = RandomRBFGeneratorDrift(change_speed=0.0001)
    stream.name = 'RBF$_{IM}$'
    streams.append(stream)
    
    # REAL WORLD STREAMS
    stream = FileStream(f'{this_dir}/../datasets/elec.csv')
    stream.name = 'ELEC'
    streams.append(stream)
    
    stream = FileStream(f'{this_dir}/../datasets/poker.csv')
    stream.name = 'POKR'
    streams.append(stream)
    
    stream = FileStream(f'{this_dir}/../datasets/gmsc.csv')
    stream.name = 'GMSC'
    streams.append(stream)
    
#    f = open('res_tex_acc.txt', 'a+')
#    f.write('\\textbf{Dataset}')
#    for clf in clfs:
#        f.write(' & \\textbf{{{}}}'.format(clf.name))
#    f.close()
#    
#    f = open('res_tex_kappa.txt', 'a+')
#    f.write('\\textbf{Dataset}')
#    for clf in clfs:
#        f.write(' & \\textbf{{{}}}'.format(clf.name))
#    f.close()
#    
#    f = open('res_tex_time.txt', 'a+')
#    f.write('\\textbf{Dataset}')
#    for clf in clfs:
#        f.write(' & \\textbf{{{}}}'.format(clf.name))
#    f.close()
#    
#    for stream in streams:
#        f = open('res_tex_acc.txt', 'a+')
#        f.write('\\\\\n')
#        f.write('{}'.format(stream.name))
#        f.close()
#        f = open('res_tex_kappa.txt', 'a+')
#        f.write('\\\\\n')
#        f.write('{}'.format(stream.name))
#        f.close()
#        f = open('res_tex_time.txt', 'a+')
#        f.write('\\\\\n')
#        f.write('{}'.format(stream.name))
#        f.close()
#        for clf in clfs:
#            tuned_clf = grid_search(copy.deepcopy(clf), copy.deepcopy(stream)) if clf.search_grid is not None else copy.deepcopy(clf)
#            five_fold(copy.deepcopy(stream), tuned_clf)