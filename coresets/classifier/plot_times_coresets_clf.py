import pandas as pd
import matplotlib.pyplot as plt

tex_like_input = '129.36,0.86,445.29,11.32\n135.06,4.03,291.39,0.79\n134.58,3.75,442.31,12.02\n124.92,0.83,275.74,4.04\n395.64,12.19,183.49,1.36\n160.65,1.10,174.14,1.57\n7.39,0.06,7.77,0.04\n17.23,0.21,44.98,0.54\n46.55,0.10,53.45,0.39\n398.74,10.81,403.27,6.22\n144.26,0.85,150.48,1.33\n'
lines = tex_like_input.split('\n')

algos = [
        'ARSLVQ$_{mean}$',
        'ARSLVQ$_{std}$',
        'MEB-Classification$_{mean}$',
        'MEB-Classificaiton$_{std}$',

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
        lines[i][j] = int(lines[i][j])
        
# build dicts like {"Name": ['ARF_mean', ...], "Projected": [1, 2, 3], "Original": [2, 4, 5, ...]}
for i in range(len(lines)):
    plt_dict = {'Algorithm': [], 'original': [], 'projected': []}
    for j in range(len(algos)):
        # input strcture is crap, hard code
        plt_dict['Algorithm'].append(algos[j])        
    for idx in [0, 1, 4, 5, 8, 9, 12, 13]:
        plt_dict['original'].append(lines[i][idx])
        plt_dict['projected'].append(lines[i][idx + 2])
        
    df = pd.DataFrame(plt_dict)
    df.set_index('Algorithm').plot(kind="bar", align='center', width=0.8)
    plt.ylabel('Runtime in seconds')
    plt.title(streams[i])
    plt.tick_params(rotation=20)
    plt.yscale('log')
    plt.savefig(f'plt/rt_rp_{streams[i].replace("$", "")}.eps')
