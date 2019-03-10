import os
import sys
import glob
import h5py
import numpy as np
import drama as drm
import scipy.io as sio

ii = int(sys.argv[1])
nn = int(sys.argv[2])

dir_add = './'+sys.argv[0][:-3]+'_res/'
drm.ch_mkdir(dir_add)

fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]

if os.path.exists(dir_add+file_names[ii]+'_'+str(nn)+'.npy'):
    exit()

print file_names[ii]    

result = []
lof_all = np.zeros(3)
ifr_all = np.zeros(3)

try:
    data = sio.loadmat(fils[ii])
    X = data['X'].astype(float)
    y = data['y'].astype(float)

except:
    data = h5py.File(fils[ii])
    X = np.array(data['X']).T.astype(float)
    y = np.array(data['y']).T.astype(float)
    
res = drm.unsupervised_outlier_finder_all(X)
arr,drts,metrs = drm.result_array(res,y,'real')

df = drm.sk_check(X,X,y,[1])
for j,scr in enumerate(['AUC','MCC','RWS']):
    lof_all[j] = df[scr][0]
    ifr_all[j] = df[scr][1]

drm.save(dir_add+file_names[ii]+'_'+str(nn),[ifr_all,lof_all,arr,drts,metrs])




