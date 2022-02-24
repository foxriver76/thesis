import pandas as pd
import matplotlib.pyplot as plt

tex_like_input = '974 , 91, 658 , 130, 39 , 1, 17 , 0.1, 9537 , 5118, 793 , 508, 18748 , 584, 186 , 5\n1094 , 60, 590 , 95, 40 , 1, 195 , 2, 9613 , 5098, 792 , 515, 18748 , 584, 195 , 2\n719 , 12, 1573 , 28, 109 , 1, 70 , 1, 9513 , 5135, 793 , 516, 17005 , 27, 597 , 1\n675 , 45., 1583 , 148, 109 , 1, 70 , 1, 9434 , 5075, 786 , 509, 17005 , 27, 597 , 1\n5749 , 400, 690 , 44, 455 , 13, 238 , 1, 2565 , 90, 167 , 5, 22863 , 310, 402 , 2\n6546 , 241, 22 , 1, 326 , 1, 225 , 1, 883 , 13, 22 , 1, 23491 , 924, 1290 , 3'
lines = tex_like_input.split('\n')

algos = [
        'ARF$_{mean}$',
        'ARF$_{std}$',
        'ARSLVQ$_{mean}$',
        'ARSLVQ$_{std}$',
        'SAM-KNN$_{mean}$',
        'SAM-KNN$_{std}$',
        'RRSLVQ$_{mean}$',
        'RRSLVQ$_{std}$',
        ]

streams = [
        'SEA$_G$', 
        'SEA$_A$',
        'LED$_G$',
        'LED$_A$',
        'NSDQ$_{tf_idf}$', 
        'NSDQ$_{embedd}$',
        ]

for i in range(len(lines)):
    lines[i] = lines[i].split(',')
    for j in range(len(lines[i])):
        lines[i][j] = float(lines[i][j].strip())
        
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
    plt.savefig(f'plt/rt_pca_{streams[i].replace("$", "")}.eps')