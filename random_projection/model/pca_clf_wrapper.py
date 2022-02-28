# -*- coding: utf-8 -*-
import numpy as np
from random_projection.model.inc_pca import IncPCA

class PCAClassifier():
    
    def __init__(self, clf, high_dims, n_components, proj=True):
        self.clf = clf
        self.initialized = False
        self.high_dims = high_dims
        if proj is True:
            self.pca = IncPCA(n_components=n_components, forgetting_factor=1)
        self.proj = proj

    def partial_fit(self, X, y, classes=None):
        if self.initialized is False:
            self.n_rand_dims = self.high_dims - X.shape[1]
            self.initialized = True
        
        X = self.enhance_data(X)
        
        if self.proj is True:
            self.pca.partial_fit(X)
            X = self.pca.transform(X)
        
        self.clf.partial_fit(X, y, classes)

    def predict(self, x):
        x = self.enhance_data(x)
        if self.proj is True:
            x = self.pca.transform(x)
            
        return self.clf.predict(x)    
        
    def enhance_data(self, X):
        X_new = np.empty([X.shape[0], self.high_dims])
        
        for i in range(X.shape[0]):
            X_new[i] = np.append(X[i], np.random.randint(2, size=self.n_rand_dims)).reshape(1, X.shape[1] + self.n_rand_dims).ravel()      

        return X_new