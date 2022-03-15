import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tex_like_input = '341.13, 1.03, 165.67, 0.48, 431.41, 1.12\n349.95, 1.61, 406.21, 0.92, 659.42, 1.2\n129.19, 0.27, 147.15, 0.17, 308.3, 0.53\n120.28, 0.72, 454.18, 0.69, 454.18, 0.69\n115.35, 0.58, 277.92, 2.09, 277.92, 2.09\n117.2, 0.6, 450.75, 0.41, 450.75, 0.41\n111.3, 0.4, 249.67, 0.33, 249.67, 0.33\n146.79, 0.17, 160.73, 0.31, 160.73, 0.31\n6.79, 0.06, 7.51, 0.02, 7.51, 0.02\n15.52, 0.05, 45.6, 0.09, 45.6, 0.09\n41.82, 0.07, 47.05, 0.15, 47.05, 0.15\n7.14, 0.63, 70.13, 1.42, 70.13, 1.42\n13.4, 0.61, 74.76, 1.24, 77.48, 1.57\n37.03, 1.87, 89.7, 1.27, 83.7, 0.9'
lines = tex_like_input.split('\n')

algos = [
        'ARSLVQ$_{mean}$',
        'ARSLVQ$_{std}$',
        'MEB-Classification$_{mean}$',
        'MEB-Classificaiton$_{std}$',
        'MCCVM$_{mean}$',
        'MCCVM$_{std}$'
        ]

streams = [
        'LED', 
        'POKER',
        'M. Squares',
        'MIXED$_A$',
        'SEA$_A$',
        'MIXED$_G$', 
        'SEA$_G$',
        'AGRAWAL', 
        'Weather',
        'Electricity',
        'GMSC',
        'OberA$_{binary}$',
        'OberA$_{compressed}$',
        'OberA$_{high}$',
        ]

for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    for j in range(len(lines[i])):
        lines[i][j] = float(lines[i][j])

data = np.array(lines)
x = np.arange(len(streams))
width = 0.8

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 3, data[:, 0], width / 3, label='ARSLVQ$_{mean}$')
rects2 = ax.bar(x + width / 3, data[:, 2], width / 3, label='MEB$_{mean}$')
rects3 = ax.bar(x, data[:, 4], width /3, label='MCCVM$_{mean}$')

ax.set_xticks(x)
ax.set_xticklabels(streams)
ax.tick_params(rotation=50)
ax.set_ylabel('Runtime in seconds')
ax.legend()
plt.savefig(f'plt/rt_clf_mc_mean.eps')
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 3, data[:, 1], width / 3, label='ARSLVQ$_{std}$')
rects2 = ax.bar(x + width / 3, data[:, 3], width / 3, label='MEB$_{std}$')
rects3 = ax.bar(x, data[:, 5], width / 3, label='MCCVM$_{std}$')

ax.set_xticks(x)
ax.set_xticklabels(streams)
ax.tick_params(rotation=50)
ax.set_ylabel('Runtime in seconds')
ax.legend()
plt.savefig(f'plt/rt_clf_mc_std.eps')
plt.show()
  
# build dicts like {"Name": ['ARF_mean', ...], "Projected": [1, 2, 3], "Original": [2, 4, 5, ...]}
#for i in range(len(lines)):
#    plt_dict = {'Algorithm': [], 'Runtime': []}
#    for j in range(len(algos)):
#        # input strcture is crap, hard code
#        plt_dict['Algorithm'].append(algos[j]) 
#        plt_dict['Runtime'].append(lines[i][j])
#        
#    df = pd.DataFrame(plt_dict)
#    df.set_index('Algorithm').plot(kind="bar", align='center', width=0.8)
#    plt.ylabel('Runtime in seconds')
#    plt.title(streams[i])
#    plt.tick_params(rotation=20)
#    plt.yscale('linear')
#    plt.savefig(f'plt/rt_clf_{streams[i].replace("$", "")}.eps')
