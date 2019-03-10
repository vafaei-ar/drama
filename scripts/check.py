import matplotlib as mpl
mpl.use('agg')

import os
import sys
import glob

dir_add = './real_data_semi_supervised_res/'

fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]

for ii in range(len(fils)):
    for n_train in range(1,21):
        for nn in range(10):
            if not os.path.exists(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn)+'.npy'):
                print (str(ii)+'_'+str(n_train)+'_'+str(nn))
                
                
                
                
                

