import pandas as pd
import matplotlib.pyplot as plt

tex_like_input = '9591, 1007, 3547, 139, 1291, 0, 1265, 1, 5473, 6986, 1286, 1301,  181717, 3804, 30431, 465\n10055, 925, 3567, 75, 1269, 1, 1260, 1, 5971, 9009, 1409, 1734, 184298, 3392, 30022, 432\n6890, 726, 2660, 211, 1247, 0, 871, 1, 8999, 2313, 3511, 1037,  183374, 3708,29939, 417\n6986, 708, 2656, 217, 1229, 1, 860, 0, 7448, 2275, 3668, 571, 185256, 4037 , 29934, 482\n12213, 374, 5705, 221, 488, 1, 400, 2, 5187, 34, 864, 10, 19361, 84, 8449, 34 \n14988, 701, 5470, 201, 353, 54, 311, 1, 1669, 82, 311, 1, 19911, 1600, 18335, 570'
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