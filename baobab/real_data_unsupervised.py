
import matplotlib as mpl
mpl.use('agg')

import drama as drm
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec

import os
import sys
import glob
import h5py
import scipy.io as sio

fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]

i = int(sys.argv[1])
j = int(sys.argv[2])

drama_all = []
lof_all = np.zeros(3)
ifr_all = np.zeros(3)

print file_names[i]


if os.path.isfile('./outputs/uns_drama_'+file_names[i]+'_'+str(j)):
    exit()

cond = True
while cond:

    try:
        try:
            data = sio.loadmat(fils[i])
            X = data['X'].astype(float)
            y = data['y'].astype(float)

        except:
            data = h5py.File(fils[i])
            X = np.array(data['X']).T.astype(float)
            y = np.array(data['y']).T.astype(float)
            
        res = drm.unsupervised_outlier_finder_all(X)
        arr,drts,metrs = drm.result_array(res,y,'real')
        drama_all.append(arr)

        df = drm.sk_check(X,X,y,[1])
        for k,scr in enumerate(['AUC','MCC','RWS']):
            lof_all[k] = df[scr][0]
            ifr_all[k] = df[scr][1]
            
        drama_all = np.array(drama_all)

        np.save('./outputs/uns_drama_'+file_names[i]+'_'+str(j),drama_all)
        np.save('./outputs/uns_lof_'+file_names[i]+'_'+str(j),lof_all)
        np.save('./outputs/uns_ifr_'+file_names[i]+'_'+str(j),ifr_all)
        
        cond = False
    except:
        pass
