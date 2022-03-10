# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv('data/obera.csv', index_col=0)

data = data.drop(columns='DateTime')

data.to_csv('data/obera_pre.csv', index=False)