# create a high dim dummy stream for mem analysis

from skmultiflow.data import SEAGenerator
import numpy as np
import pandas as pd

stream = SEAGenerator()
stream.next_sample()

RANGE = 10000
DIM = 52
n_rand_dims = DIM - stream.current_sample_x.size
multiply = n_rand_dims // stream.current_sample_x.size

data = []
labels = []

for i in range(RANGE):
    current_sample_x = np.array([[]])
    for _m in range(multiply):
        current_sample_x = np.concatenate(
                (current_sample_x, stream.current_sample_x), axis=1)
    data.append(current_sample_x.ravel())
    labels.append(stream.current_sample_y.ravel()[0])
    stream.next_sample()
    
df = pd.DataFrame(data)
df['labels'] = labels

df.to_csv('high_dim_stream.csv')