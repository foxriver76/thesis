# -*- coding: utf-8 -*-

import sys,os
sys.path.append(os.path.abspath(__file__ + "/../../"))
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential
from random_projection.model.pca_clf_wrapper import PCAClassifier
from skmultiflow.prototype import RobustSoftLearningVectorQuantization as ARSLVQ
from skmultiflow.meta import AdaptiveRandomForestClassifier as ARF
from skmultiflow.lazy import SAMKNNClassifier as SAMKNN
from random_projection.model.rrslvq import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ

N_SAMPLES = 10000

stream = SEAGenerator()

clf_rslvq = PCAClassifier(clf=ARSLVQ(), n_components=48, high_dims=250, proj=False)
clf_rrslvq = PCAClassifier(clf=RRSLVQ(), n_components=48, high_dims=250, proj=False)
clf_arf = PCAClassifier(clf=ARF(), n_components=48, high_dims=250, proj=False)
clf_samknn = PCAClassifier(clf=SAMKNN(), n_components=48, high_dims=250, proj=False)

evaluator = EvaluatePrequential(metrics=['model_size', 'accuracy'], 
                                max_samples=N_SAMPLES, show_plot=False,
                                output_file='_pca_mem_no_proj.csv', 
                                batch_size=50)

evaluator.evaluate(stream, [clf_rslvq, clf_rrslvq, clf_arf, clf_samknn], model_names=[
        'ARSLVQ',
        'RRSLVQ',
        'ARF',
        'SAM-KNN'
        ])


