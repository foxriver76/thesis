# -*- coding: utf-8 -*-

from skmultiflow.data import SEAGenerator
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential

stream = SEAGenerator()

clf = NaiveBayes()

evaluator = EvaluatePrequential(metrics=['model_size', 'accuracy'])

evaluator.evaluate(stream, clf)