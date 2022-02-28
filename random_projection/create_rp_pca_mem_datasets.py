# -*- coding: utf-8 -*-
import numpy as np
from skmultiflow.data import SEAGenerator

def enhance_data(X, high_dims):
    X_new = np.empty([X.shape[0], high_dims])
    n_rand_dims = high_dims - X.shape[1]
    
    for i in range(X.shape[0]):
        X_new[i] = np.append(X[i], np.random.randint(2, size=n_rand_dims)).reshape(1, X.shape[1] + n_rand_dims).ravel()      

    return X_new


def create_data(N_SAMPLES, HIGH_DIMS, path):
    stream = SEAGenerator()
    
    X, y = stream.next_sample(N_SAMPLES)
    
    X = enhance_data(X, HIGH_DIMS)
    data = np.hstack((X, y.reshape(-1,1)))
    
    print(data.shape)
    
    data.tofile(path, sep=',')
    

create_data(10000, 10000, '../datasets/rp_mem_sea.csv')
create_data(10000, 250, '../datasets/pca_mem_sea.csv')