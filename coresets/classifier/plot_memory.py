# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

colors = {
        'MEB': 'blue',
        'ARSLVQ': 'orange'
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

# mixed abrupt
plot_memory('_mixed_a_memory_other.csv', 'plt/coreset_mem_mixed_a.eps')
# mixed gradual
plot_memory('_mixed_g_memory_other.csv', 'plt/coreset_mem_mixed_g.eps')