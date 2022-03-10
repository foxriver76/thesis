import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tex_like_input = '129.36,0.86,445.29,11.32\n135.06,4.03,291.39,0.79\n134.58,3.75,442.31,12.02\n124.92,0.83,275.74,4.04\n395.64,12.19,183.49,1.36\n160.65,1.10,174.14,1.57\n7.39,0.06,7.77,0.04\n17.23,0.21,44.98,0.54\n46.55,0.10,53.45,0.39\n398.74,10.81,403.27,6.22\n144.26,0.85,150.48,1.33'
lines = tex_like_input.split('\n')

algos = [
        'ARSLVQ$_{mean}$',
        'ARSLVQ$_{std}$',
        'MEB-Classification$_{mean}$',
        'MEB-Classificaiton$_{std}$'
        ]

streams = [
        'MIXED$_A$', 
        'SEA$_A$',
        'MIXED$_G$', 
        'SEA$_G$',
        'LED',
        'AGRAWAL', 
        'Weather',
        'Electricity',
        'GMSC',
        'POKER',
        'Moving Squares'
        ]

for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    for j in range(len(lines[i])):
        lines[i][j] = float(lines[i][j])

data = np.array(lines)
x = np.arange(len(streams))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, data[:, 0], width, label='ARSLVQ$_{mean}$')
rects2 = ax.bar(x + width / 2, data[:, 2], width, label='MEB$_{mean}$')

ax.set_xticks(x)
ax.set_xticklabels(streams)
ax.tick_params(rotation=35)
ax.set_ylabel('Runtime in seconds')
ax.legend()
plt.savefig(f'plt/rt_clf_mean.eps')
plt.show()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width / 2, data[:, 1], width, label='ARSLVQ$_{std}$')
rects2 = ax.bar(x + width / 2, data[:, 3], width, label='MEB$_{std}$')

ax.set_xticks(x)
ax.set_xticklabels(streams)
ax.tick_params(rotation=35)
ax.set_ylabel('Runtime in seconds')
ax.legend()
plt.savefig(f'plt/rt_clf_std.eps')
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
