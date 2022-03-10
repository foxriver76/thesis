# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv('../../datasets/obera.csv', index_col=0)

data = data.drop(columns='DateTime')
data = data.drop(columns='Phase_compressed')
data = data.drop(columns='Production')

data.to_csv('../../datasets/obera_high.csv', index=False)