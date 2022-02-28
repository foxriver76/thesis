from sklearn.random_projection import SparseRandomProjection
import numpy as np

class RandomProjectionClassifier():
    
    def __init__(self, clf, n_components, high_dims=10000, proj=True):
        self.clf = clf
        self.initialized = False
        if proj is True:
            self.rp = SparseRandomProjection(n_components, density='auto')
        self.high_dims = high_dims
        self.proj = proj
        
    def partial_fit(self, X, y, classes=None):
        if self.initialized is False:
            self.n_rand_dims = self.high_dims - X.shape[1]
        
        X = self.enhance_data(X)
        
        if self.initialized is False:
            # create rp
            if self.proj is True:
                self.rp.fit(X)
            self.initialized = True
        
        if self.proj is True:
            X = self.rp.transform(X)

        self.clf.partial_fit(X, y, classes)
        
    def predict(self, x):
        x = self.enhance_data(x)
        if self.proj is True:
            x = self.rp.transform(x)
        return self.clf.predict(x)
    
    def enhance_data(self, X):
        X_new = np.empty([X.shape[0], self.high_dims])
        
        for i in range(X.shape[0]):
            X_new[i] = np.append(X[i], np.random.randint(2, size=self.n_rand_dims)).reshape(1, X.shape[1] + self.n_rand_dims).ravel()      

        return X_new