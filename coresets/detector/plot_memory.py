# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

colors = {
        'K-MEBWIND': 'blue',
        'MEBWIND': 'red',
        'ADWIN': 'green',
        'KSWIN': 'orange'
        }

lines = {
        'K-MEBWIND': 'dotted',
        'MEBWIND': 'solid',
        'ADWIN': 'dashed',
        'KSWIN': 'dashdot'
        }

marker = {
        'K-MEBWIND': 'o',
        'MEBWIND': '',
        'ADWIN': 'D',
        'KSWIN': 4
        }


def plot_memory(input_path, output_path):
    df = pd.read_csv(input_path)
    
    for dim in df:
        if dim.startswith('model_size'):
            label = dim.split('[')[1].split(']')[0]
            plt.plot(df['id'].values, df[dim].values, label=label, 
                     c=colors[label], linestyle=lines[label], marker=marker[label])
    
    plt.legend(loc='best')
    plt.yscale('log')
    plt.xlabel('Timestep $t$')
    plt.ylabel('Model size in kB')   
    plt.savefig(output_path)
    plt.show()

# led
plot_memory('_SEA_high_memory_other.csv', 'plt/coreset_cdd_memory.eps')
