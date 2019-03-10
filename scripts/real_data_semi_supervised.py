import matplotlib as mpl
mpl.use('agg')

import os
import sys
import glob
import h5py
import numpy as np
import pylab as plt
import drama as drm
import scipy.io as sio
from matplotlib import gridspec

ii = int(sys.argv[1])
n_train = int(sys.argv[2])
nn = int(sys.argv[3])

dir_add = './'+sys.argv[0][:-3]+'_res/'
drm.ch_mkdir(dir_add)

fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]

if os.path.exists(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn)+'.npy'):
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
    
iinds = np.argwhere(y[:,0]==0)[:,0]
oinds = np.argwhere(y[:,0]==1)[:,0]
nhalf = iinds.shape[0]//2

if oinds.shape[0]<=n_train:
    np.save(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn),[np.nan,np.nan,np.nan])
    exit()

np.random.shuffle(iinds)
np.random.shuffle(oinds)

X_train = np.concatenate([X[iinds[:nhalf]],X[oinds[:n_train]]],axis=0)
y_train = np.concatenate([y[iinds[:nhalf]],y[oinds[:n_train]]],axis=0)
X_test = np.concatenate([X[iinds[:]],X[oinds[n_train:]]],axis=0)
y_test = np.concatenate([y[iinds[:]],y[oinds[n_train:]]],axis=0)

X_train = X_train/X_train.max()
X_test = X_test/X_test.max()

df = drm.sk_check(X_test,X_test,y_test,[1])

res = drm.unsupervised_outlier_finder_all(X_train)        

auc = []
mcc = []
rws = []

auc_b = -100
mcc_b = -100
rws_b = -100

for i in range(50):
    for j in ['real','latent']:
        o1 = res[j][i]
        auc = drm.roc_auc_score(y_train==1, o1)
        mcc = drm.MCC(y_train==1, o1)
        rws = drm.rws_score(y_train==1, o1)

        if auc_b<auc:
            auc_b = auc
            auc_set = [j,res['drt'][i],res['metric'][i]]

        if mcc_b<mcc:
            mcc_b = mcc
            mcc_set = [j,res['drt'][i],res['metric'][i]]

        if rws_b<rws:
            rws_b = rws
            rws_set = [j,res['drt'][i],res['metric'][i]]


res = drm.get_outliers(X_test,auc_set[1],auc_set[2],clustering=None,z_dim=2,space=auc_set[0])
o1 = res[auc_set[0]][auc_set[2]]

res = drm.get_outliers(X_test,mcc_set[1],mcc_set[2],clustering=None,z_dim=2,space=mcc_set[0])
o2 = res[mcc_set[0]][mcc_set[2]]

res = drm.get_outliers(X_test,rws_set[1],rws_set[2],clustering=None,z_dim=2,space=rws_set[0])
o3 = res[rws_set[0]][rws_set[2]]

acc = drm.roc_auc_score(y_test==1, o1)
mcc = drm.MCC(y_test==1, o2)
rws = drm.rws_score(y_test==1, o3)
print(acc,mcc,rws)

drm.save(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn),[acc,mcc,rws,df])


