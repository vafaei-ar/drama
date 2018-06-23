
import matplotlib as mpl
mpl.use('agg')

import drama as drm
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
from sklearn.metrics import roc_auc_score

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
frac = float(0.5)

drama_all = []
lof_all = np.zeros(3)
ifr_all = np.zeros(3)

print file_names[i]

try:
    data = sio.loadmat(fils[i])
    X = data['X'].astype(float)
    y = data['y'].astype(float)

except:
    data = h5py.File(fils[i])
    X = np.array(data['X']).T.astype(float)
    y = np.array(data['y']).T.astype(float)
          
i_ind = np.argwhere((y == 0))[:,0]
o_ind = np.argwhere((y != 0))[:,0]

i_train,i_test = drm.random_choice(i_ind,frac)

train_idx = i_train
test_idx = np.concatenate([i_test,o_ind])
                                            
X_train = X[train_idx]   
y_train = y[train_idx]
                                            
X_test = X[test_idx]
y_test = y[test_idx]
                 
res = drm.novelty_finder_all(X_train,X_test,n_slpit=4)
arr,drts,metrs = drm.result_array(res,y_test,'real')
drama_all.append(arr)

df = drm.sk_check(X,X,y,[1])
for j,scr in enumerate(['AUC','MCC','RWS']):
    lof_all[i,j] = df[scr][0]
    ifr_all[i,j] = df[scr][1]

drama_all = np.array(drama_all)

np.save('./outputs/nov_drama_'+file_names[i]+'_'+str(j),drama_all)
np.save('./outputs/nov_lof_'+file_names[i]+'_'+str(j),lof_all)
np.save('./outputs/nov_ifr_'+file_names[i]+'_'+str(j),ifr_all)

