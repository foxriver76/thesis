#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:52:44 2020

@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from skmultiflow.bayes.naive_bayes import NaiveBayes
from bix.detectors.kswin import KSWIN
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.eddm import  EDDM
from skmultiflow.drift_detection.ddm import DDM
from bix.detectors.ksvec import KSVEC
from sw_mebwin import SWMEBWIN as MEBWIND
from kernel_swmebwin import Kernel_SWMEBWIN as KMEBWIND

class cdnb(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=0.001, drift_detector="KSWIN"):
        self.classifier = NaiveBayes()
        self.init_drift_detection = True
        self.drift_detector = drift_detector.upper()
        self.confidence = alpha
        self.n_detections = 0

    def partial_fit(self, X, y, classes=None):
            """
            Calls the MultinomialNB partial_fit from sklearn.
            ----------
            x : array-like, shape = [n_samples, n_features]
              Training vector, where n_samples in the number of samples and
              n_features is the number of features.
            y : array, shape = [n_samples]
              Target values (integers in classification, real numbers in
              regression)
            Returns
            --------
            """
            if self.concept_drift_detection(X, y, classes):
                self.classifier.reset()

            self.classifier.partial_fit(X, y, classes)
            return self

    def predict(self, X):
        return self.classifier.predict(X)

    def concept_drift_detection(self, X, Y, classes):
        if self.init_drift_detection:
            if self.drift_detector == "KSWIN":
                self.cdd = [KSWIN(w_size = 100, stat_size = 30, alpha=self.confidence) for elem in X.T]
            if self.drift_detector == "ADWIN":
                self.cdd = [ADWIN() for elem in X.T]
            if self.drift_detector == "DDM":
                self.cdd = [DDM() for elem in X.T]
            if self.drift_detector == "EDDM":
                self.cdd = [EDDM() for elem in X.T]
#            if self.drift_detector == "KSVEC":
#                self.cdd = KSVEC(vec_size=X.shape[1])
            if self.drift_detector == 'MEBWIND':
                self.cdd = MEBWIND(classes=classes, w_size=80, epsilon=0.05)
            if self.drift_detector == 'KMEBWIND':
                self.cdd = KMEBWIND(classes=classes, w_size=80, epsilon=0.05)
                
            self.init_drift_detection = False
        self.drift_detected = False

        if not self.init_drift_detection:
#            if self.drift_detector == "KSVEC":
#                self.cdd.add_element(X)
#                if self.cdd.detected_change():
#                    self.drift_detected = True
            if self.drift_detector == 'MEBWIND' or self.drift_detector == 'KMEBWIND':
                for i in range(len(Y)):
                    self.cdd.add_element(X[i], Y[i])

                    if self.cdd.detected_change:
                        self.drift_detected = True
                        self.n_detections += 1
            else:
                for elem, detector in zip(X.T, self.cdd):
                    for e in elem:
                        detector.add_element(e)
                        if detector.detected_change():
                            self.drift_detected = True
                            self.n_detections = self.n_detections +1

        return self.drift_detected

if __name__ == '__main__':
    from skmultiflow.data import FileStream
    from skmultiflow.evaluation import EvaluatePrequential
    
    streams = [
#            FileStream('../datasets/elec.csv'),
            FileStream('../datasets/poker.csv'),
#            FileStream('../datasets/weather.csv')
            ]
    # GMSC
    
#    streams[0].name = 'Electricity'
    streams[0].name = 'Weather'

#    streams[1].name = 'POKER'
#    streams[2].name = 'Weather'
    
    for stream in streams:
        
        models = [
                cdnb(drift_detector='KSWIN'),
                cdnb(drift_detector='ADWIN'),
                cdnb(drift_detector='MEBWIND'),
                cdnb(drift_detector='KMEBWIND')
                ]
    
        
        evaluator = EvaluatePrequential(max_samples=10000000, show_plot=True)
        evaluator.evaluate(stream, models, model_names=['KSWIN', 'ADWIN', 'MEBWIND', 'KMEBWIND'])
        
        print(f'{stream.name}:')
        print(f'KSWIN: {models[0].n_detections}')
        print(f'ADWIN: {models[1].n_detections}')
        print(f'MEBWIND: {models[2].n_detections}')
        print(f'KMEBWIND: {models[3].n_detections}')
        
#Processed samples: 45312
#Mean performance:
#KSWIN - Accuracy     : 0.8340
#KSWIN - Kappa        : 0.6560
#ADWIN - Accuracy     : 0.8346
#ADWIN - Kappa        : 0.6614
#MEBWIND - Accuracy     : 0.8456
#MEBWIND - Kappa        : 0.6840
#KMEBWIND - Accuracy     : 0.8535
#KMEBWIND - Kappa        : 0.6992
#Electricity:
#KSWIN: 2790
#ADWIN: 238
#MEBWIND: 234
#KMEBWIND: 276
#Prequential Evaluation
#Evaluating 1 target(s).
#Pre-training on 200 sample(s).
#Evaluating...
# #################### [100%] [9829.50s]
#Processed samples: 829201
#Mean performance:
#KSWIN - Accuracy     : 0.7621
#KSWIN - Kappa        : 0.5648
#ADWIN - Accuracy     : 0.7470
#ADWIN - Kappa        : 0.5471
#MEBWIND - Accuracy     : 0.7644
#MEBWIND - Kappa        : 0.5777
#KMEBWIND - Accuracy     : 0.7652
#KMEBWIND - Kappa        : 0.5783
#POKER:
#KSWIN: 33480
#ADWIN: 7948
#MEBWIND: 5479
#KMEBWIND: 5901
        
#        Processed samples: 18159
#Mean performance:
#KSWIN - Accuracy     : 0.6982
#KSWIN - Kappa        : 0.2598
#ADWIN - Accuracy     : 0.7195
#ADWIN - Kappa        : 0.3274
#MEBWIND - Accuracy     : 0.7239
#MEBWIND - Kappa        : 0.3632
#KMEBWIND - Accuracy     : 0.7185
#KMEBWIND - Kappa        : 0.3490
#Weather:
#KSWIN: 1488
#ADWIN: 1097
#MEBWIND: 128
#KMEBWIND: 132
    