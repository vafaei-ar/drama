
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

drama_all = np.zeros(3)
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
o_train,o_test = drm.random_choice(o_ind,frac)

train_idx = np.concatenate([i_train,o_train])
test_idx = np.concatenate([i_test,o_test])
                                            
X_train = X[train_idx]   
y_train = y[train_idx]
                                            
X_test = X[test_idx]
y_test = y[test_idx]
                 
o1,o2,o3 = drm.supervised_outlier_finder_all(X_train,y_train,X_test)
                                            
auc = roc_auc_score(y_test==1, o1)
drama_all[i,0] = auc
mcc = drm.MCC(y_test==1, o2)
drama_all[i,1] = mcc
rws = drm.rws_score(y_test==1, o3)           
drama_all[i,2] = rws

df = drm.sk_check(X,X,y,[1])
for k,scr in enumerate(['AUC','MCC','RWS']):
    lof_all[k] = df[scr][0]
    ifr_all[k] = df[scr][1]


np.save('./outputs/sup_drama_'+file_names[i]+'_'+str(j),drama_all)
np.save('./outputs/sup_lof_'+file_names[i]+'_'+str(j),lof_all)
np.save('./outputs/sup_ifr_'+file_names[i]+'_'+str(j),ifr_all)
