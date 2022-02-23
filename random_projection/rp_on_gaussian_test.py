#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:03:47 2019
@author: Moritz Heusinger <moritz.heusinger@gmail.com>
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn import random_projection
from random_projection.model.arslvq import RSLVQ as ARSLVQ
from sklearn.metrics import accuracy_score

n_samples = 500
## get Dataset
X, y_true = make_blobs(n_samples=n_samples, centers=15,
                       cluster_std=1.0, n_features=10000, random_state=None)

y_true = y_true.reshape(y_true.size, 1)

## Get good number for reduction
good_n = random_projection.johnson_lindenstrauss_min_dim(n_samples=X[0].size, eps=0.3)
print('Number of dimensions by Lindenstrauss Lemma: {}'.format(good_n))

# n_components are the target dimensions
transformer = random_projection.SparseRandomProjection(n_components=good_n, density=1/3)
#transformer = random_projection.GaussianRandomProjection(n_components=good_n)

rslvq_proj = ARSLVQ(gradient_descent='Adadelta')
rslvq_orig = ARSLVQ(gradient_descent='Adadelta')

#test_one_dp = [X[0, :]]
#X_test_one_dp = transformer.fit_transform(test_one_dp)
#X_test_all = transformer.fit_transform(X)
#
#y_pred = RSLVQ().fit(X_test_all, y_true).predict(X_test_one_dp)

#acc = accuracy_score(y_true, y_pred)

# stream mock
X_transformed  = np.zeros(n_samples - 1)
y_proj_sum = np.zeros(n_samples - 1)
y_orig_sum = np.zeros(n_samples - 1)

classes = np.unique(y_true).tolist()

for i in range(n_samples):
    # Print Progression every 5 %
    if i > 0 and i % (n_samples / 20) == 0:
        print('{} %'.format(int(i / n_samples * 100)))
              
    dp = X[i, :].reshape(1, X[i].size)
    
    ## create random matrix only at the beginning
    if i == 0:
        transformed_dp = transformer.fit_transform(dp)
        ## pretrain fit current tuple - on first dp, because we cannot predict with unlearned model  
        rslvq_proj.partial_fit(transformed_dp, y_true[i], classes=classes)
        rslvq_orig.partial_fit(dp, y_true[i], classes=classes)
        # pretrain happened so lets skip the rest of the exceution to start predicting
        continue 
    else:
        transformed_dp = transformer.transform(dp)

    ## save the projected dp to a new array
    #X_transformed[i - 1] = transformed_dp
    
    ## predict with proj and orig rslvq + data
    y_proj = rslvq_proj.predict(transformed_dp)
    y_orig = rslvq_orig.predict(dp)
    
    ## append predicted labels
    y_proj_sum[i - 1] = y_proj
    y_orig_sum[i - 1] = y_orig
    
    ## fit current tuple    
    rslvq_proj.partial_fit(transformed_dp, y_true[i], classes=classes)
    rslvq_orig.partial_fit(dp, y_true[i], classes=classes)
    
## calculate accuracy - first label was pretrain so we have predicted one less
acc_proj = accuracy_score(y_proj_sum, y_true[1:])
acc_orig = accuracy_score(y_orig_sum, y_true[1:])

print(f'Accuracy original space: {acc_orig}')
print(f'Accuracy projected space: {acc_proj}')
