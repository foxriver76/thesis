# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../../datasets/obera.csv', index_col=0)

data['Phase'] = data['Phase'] - 1 #make label start at 0
data['Phase_compressed'] = data['Phase_compressed'] - 1

le = LabelEncoder()
# we have missing labels
data['Phase'] = le.fit(data['Phase']).transform(data['Phase'])

data = data.drop(columns='DateTime')
data = data.drop(columns='Phase_compressed')
data = data.drop(columns='Production')

data.to_csv('../../datasets/obera_high.csv', index=False)