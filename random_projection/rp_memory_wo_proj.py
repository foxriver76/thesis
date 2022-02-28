# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential
from random_projection.model.random_proj_clf_wrapper import RandomProjectionClassifier
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as ARSLVQ
from skmultiflow.meta import AdaptiveRandomForestClassifier as ARF
from skmultiflow.lazy import SAMKNNClassifier as SAMKNN
from random_projection.model.rrslvq import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ

N_SAMPLES = 10000

stream = SEAGenerator()

clf_rslvq = RandomProjectionClassifier(clf=ARSLVQ(), n_components=1000, high_dims=10000, proj=False)
#clf_rrslvq = RandomProjectionClassifier(clf=RRSLVQ(), n_components=1000, high_dims=10000, proj=False)
clf_arf = RandomProjectionClassifier(clf=ARF(), n_components=1000, high_dims=10000, proj=False)
clf_samknn = RandomProjectionClassifier(clf=SAMKNN(), n_components=1000, high_dims=10000, proj=False)

evaluator = EvaluatePrequential(metrics=['model_size', 'accuracy'], 
                                max_samples=N_SAMPLES, show_plot=False,
                                output_file='_rp_mem_no_proj.csv')

evaluator.evaluate(stream, [clf_rslvq, clf_arf, clf_samknn], model_names=[
        'ARSLVQ',
#        'RRSLVQ',
        'ARF',
        'SAM-KNN'
        ])