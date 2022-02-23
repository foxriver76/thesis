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
from random_projection.model.rff_base import Base as RFF
from random_projection.model.rrslvq import ReactiveRobustSoftLearningVectorQuantization as RRSLVQ
from random_projection.model.rslvq import RSLVQ
import copy
import time
res_file = "rp_nasdaq.txt"
def next_sample():
    line = f.readline()
    if line == '': return 'EOF', 'EOF'
    arr = np.array(line.split(','), dtype='float64')
    # 0 column is index -> drop, 1 col label
    return np.array([arr[2:]]), np.array([arr[1]])

clfs = [RSLVQ(prototypes_per_class=2,gradient_descent="Adadelta"),RRSLVQ(prototypes_per_class=2,confidence=1e-10),ARF(), SAMKNN()]
    
classes = np.arange(0, 15, 1)

transformer = SparseRandomProjection(n_components=1000)

"""TF-IDF"""
#High dim
for clf in clfs:
   acc_fold = []
   kappa_fold = []
   time_fold = []
   
   for _ in range(3):
       _clf = copy.deepcopy(clf)
       f = open('../dataset/nasdaq_stream_wo_sentiment.csv')
       start_time = time.time()
       y_true = []
       y_pred = []

       x, y = next_sample()
       #print(f'{x.shape[1]} dimensions')
       _clf.partial_fit(x, y, classes=classes.tolist())
       
       while True:
           x, y = next_sample()
           if type(x) == str and x == 'EOF': break
           # test with less labels
       #    if y > 9: continue
           y_pred.append(_clf.predict(x)[0])
       
           y_true.append(y[0])
           # update clf
           _clf.partial_fit(x, y)
       
       f.close()
       print('high fold done')
       
       acc_fold.append(accuracy_score(y_true, y_pred))
       kappa_fold.append(cohen_kappa_score(y_true, y_pred))
       time_fold.append(time.time() - start_time)
       
   f = open(res_file, 'a+')
   f.write(50 * '-')
   f.write('\n')
   f.write(f'High dim test Nasdaq TFIDF: \n{_clf}\n')
   f.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
   f.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
   f.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
   f.write(50 * '-')
   f.write('\n\n')
   f.close()
        
# Low Dim
for clf in clfs:
    acc_fold = []
    kappa_fold = []
    time_fold = []
    
    for _ in range(3):
        _clf = copy.deepcopy(clf)
        f = open('../dataset/nasdaq_stream_wo_sentiment.csv')
        start_time = time.time()
        y_true = []
        y_pred = []

        x, y = next_sample()
       # print(f'{x.shape[1]} dimensions')
        # Learn RP
        x = transformer.fit_transform(x)
        _clf.partial_fit(x, y, classes=classes.tolist())
        
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
        
    fi = open(res_file, 'a+')
    fi.write(50 * '-')
    fi.write('\n')
    fi.write(f'Nasdaq TFIDF Low dim test: \n{_clf}\n')
    fi.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
    fi.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
    fi.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
    fi.write(50 * '-')
    fi.write('\n\n')
    fi.close()
    
"""SKIP-GRAM"""
fi = open(res_file, 'a+')
fi.write('SKIP-GRAM\n')
fi.close()
data = np.load('../dataset/skip-gram-embed-w-label.npy')

#HIGH-DIM
X, y = data[:, :-1], data[:, -1]

for clf in clfs:
   acc_fold = []
   kappa_fold = []
   time_fold = []
   
   for _ in range(3):
       _clf = copy.deepcopy(clf)
       start_time = time.time()
       y_true = []
       y_pred = []
       
       x = data[0, :-1].reshape(1, 1000)
       y = data[0, -1].reshape(1, 1)
       
       # pretrain
       _clf.partial_fit(x, y.ravel(), classes=classes.tolist())
       
       for i in range(data.shape[0]):
           x = data[i, :-1].reshape(1, 1000)
           y = data[i, -1].reshape(1, 1)
           y_pred.append(_clf.predict(x)[0])
       
           y_true.append(y[0])
           # update clf
           _clf.partial_fit(x, y.ravel())
       
       print('high fold skipgram done')
       
       acc_fold.append(accuracy_score(y_true, y_pred))
       kappa_fold.append(cohen_kappa_score(y_true, y_pred))
       time_fold.append(time.time() - start_time)
       
   fi = open(res_file, 'a+')
   fi.write(50 * '-')
   fi.write('\n')
   fi.write(f'High dim test skip-gram: \n{_clf}\n')
   fi.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
   fi.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
   fi.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
   fi.write(50 * '-')
   fi.write('\n\n')
   fi.close()

# LOW-DIM
X, y = data[:, :-1], data[:, -1]

for clf in clfs:
   acc_fold = []
   kappa_fold = []
   time_fold = []
   
   for _ in range(3):
       _clf = copy.deepcopy(clf)
       start_time = time.time()
       y_true = []
       y_pred = []
       
       x = data[0, :-1].reshape(1, 1000)
       y = data[0, -1].reshape(1, 1)
       
       # Learn RP
       x = transformer.fit_transform(x)
       _clf.partial_fit(x, y, classes=classes.tolist())
       
       for i in range(data.shape[0]):
           x = data[i, :-1].reshape(1, 1000)
           y = data[i, -1].reshape(1, 1)
           x = transformer.transform(x)
           y_pred.append(_clf.predict(x)[0])
       
           y_true.append(y[0])
           # update clf
           _clf.partial_fit(x, y)
       
       print('low fold skipgram done')
       
       acc_fold.append(accuracy_score(y_true, y_pred))
       kappa_fold.append(cohen_kappa_score(y_true, y_pred))
       time_fold.append(time.time() - start_time)
        
   fi = open(res_file, 'a+')
   fi.write(50 * '-')
   fi.write('\n')
   fi.write(f'Low dim test skip-gram: \n{_clf}\n')
   fi.write(f'Acc: {np.array(acc_fold).mean()} +- {np.array(acc_fold).std()}\n')
   fi.write(f'Kappa: {np.array(kappa_fold).mean()} +- {np.array(kappa_fold).std()}\n')
   fi.write(f'Time: {np.array(time_fold).mean()} +- {np.array(time_fold).std()}\n')
   fi.write(50 * '-')
   fi.write('\n\n')
   fi.close()

