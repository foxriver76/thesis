# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

colors = {
        'MEB$_5$': 'blue',
        'MEB$_{300}$': 'green',
        'MCCVM$_5$': 'purple',
        'MCCVM$_{300}$': 'red',
        'ARSLVQ': 'orange'
        }

lines = {
        'MEB$_5$': 'solid',
        'MEB$_{300}$': 'solid',
        'MCCVM$_5$': 'dashed',
        'MCCVM$_{300}$': 'dashed',
        'ARSLVQ': 'solid'
        }


def plot_memory(input_path, output_path):
    df = pd.read_csv(input_path)
    
    for dim in df:
        if dim.startswith('model_size'):
            label = dim.split('[')[1].split(']')[0]
            plt.plot(df['id'].values, df[dim].values, label=label, 
                     c=colors[label], linestyle=lines[label])
    
    plt.legend(loc='best')
    plt.yscale('linear')
    plt.xlabel('Timestep $t$')
    plt.ylabel('Model size in kB')   
    plt.savefig(output_path)
    plt.show()

# led
plot_memory('_Led Generator_memory_other.csv', 'plt/coreset_mccvm_memory.eps')
