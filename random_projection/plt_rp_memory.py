# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

colors = {
        'SAM-KNN': 'red',
        'ARSLVQ': 'blue',
        'RRSLVQ': 'orange',
        'ARF': 'green'
        }

def plot_memory(input_path, output_path):
    df = pd.read_csv(input_path)
    
    for dim in df:
        if dim.startswith('model_size'):
            label = dim.split('[')[1].split(']')[0]
            plt.plot(df['id'].values, df[dim].values, label=label, 
                     c=colors[label])
    
    plt.legend(loc='best')
    plt.yscale('log')
    plt.xlabel('Timestep $t$')
    plt.ylabel('Model size in kB')   
    plt.savefig(output_path)
    plt.show()

# no projection
plot_memory('_rp_mem_no_proj.csv', 'plt/rp_memory_no_proj.eps')
# projection
plot_memory('_rp_mem_w_proj.csv', 'plt/rp_memory_w_proj.eps')
# pca- no projection
plot_memory('_pca_mem_no_proj.csv', 'plt/pca_memory_no_proj.eps')
# pca- projection
plot_memory('_pca_mem_w_proj.csv', 'plt/pca_memory_w_proj.eps')