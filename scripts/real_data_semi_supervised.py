import matplotlib as mpl
mpl.use('agg')

import os
import sys
import glob
import h5py
import argparse
import numpy as np
#import pylab as plt
import drama as drm
import scipy.io as sio
#from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore", message='default contamination parameter 0.1 will change in version 0.22 to "auto". This will change the predict method behavior.')
warnings.filterwarnings("ignore", message='Data with input dtype float64 was converted to bool by check_pairwise_arrays.')
warnings.filterwarnings("ignore", message='Invalid value encountered in percentile')

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--id', action="store", type=int, required=True)
parser.add_argument('--ntrain', action="store", type=int, required=True)
parser.add_argument('--nn', action="store", type=int, required=True)

args = parser.parse_args()
ii = args.id
n_train = args.ntrain
nn = args.nn

dir_add = './'+sys.argv[0][:-3]+'_res/'
drm.ch_mkdir(dir_add)

fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]

if os.path.exists(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn)+'.pickle'):
    exit()
    
print(file_names[ii])

try:
    data = sio.loadmat(fils[ii])
    X = data['X'].astype(float)
    y = data['y'].astype(float)

except:
    data = h5py.File(fils[ii])
    X = np.array(data['X']).T.astype(float)
    y = np.array(data['y']).T.astype(float)
    
if n_train==0:
    dd = drm.grid_run_drama(X_seen=X,y_seen=y)
    ll = drm.grid_run_lof(X_seen=X,y_seen=y)
    ii = drm.grid_run_iforest(X_seen=X,y_seen=y)
    
else:
    X_train,y_train,X_test,y_test = drm.data_split(X,y,n_train)
    dd = drm.grid_run_drama(X_seen=X_train ,y_seen=y_train ,X_unseen=X_test, y_unseen=y_test, n_split=1)
    ll = drm.grid_run_lof(X_seen=X_train ,y_seen=y_train ,X_unseen=X_test, y_unseen=y_test)
    ii = drm.grid_run_iforest(X_seen=X_train ,y_seen=y_train ,X_unseen=X_test, y_unseen=y_test)

drm.save(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn),[dd,ll,ii])



