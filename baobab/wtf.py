import matplotlib as mpl
mpl.use('agg')

import drama as drm
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec

import os
import sys
import glob
#import h5py
#import scipy.io as sio
from scipy.ndimage import imread

from time import time

levs = 3
z_dim = 2

i = int(sys.argv[1])

def batch():
    """This fucntion can read train and test data."""
    adds = glob.glob('./train/'+'*.png')+glob.glob('./class/'+'*.png')
#     adds = random.sample(adds,n_bath)
    n_data = len(adds)
    x = np.zeros((n_data,64*64), dtype=float)
    y = np.zeros((n_data), dtype=int)

    for i,add in enumerate(adds):
        xp = imread(add)
        xp = xp.reshape(-1)
        xp = np.array(xp,dtype=float)
        x[i] = xp/xp.max()
        if 'B' in add:
            y[i] = 0
        else:
            y[i] = 1  

    return x,y

drama_all = []
lof_all = np.zeros(3)
ifr_all = np.zeros(3)

file_name = 'wtf'
if os.path.isfile('../outputs/uns_drama_'+file_name+'_'+str(i)):
    exit()

#X, y = batch()
#    
#res = drm.unsupervised_outlier_finder_all(X)
#arr,drts,metrs = drm.result_array(res,y,'real')
#drama_all.append(arr)

#df = drm.sk_check(X,X,y,[1])
#for k,scr in enumerate(['AUC','MCC','RWS']):
#    lof_all[k] = df[scr][0]
#    ifr_all[k] = df[scr][1]
#    
#drama_all = np.array(drama_all)

#np.save('../outputs/uns_drama_'+file_name+'_'+str(i),drama_all)
#np.save('../outputs/uns_lof_'+file_name+'_'+str(i),lof_all)
#np.save('../outputs/uns_ifr_'+file_name+'_'+str(i),ifr_all)

cond = True
while cond:

    try:
        X, y = batch()
            
        res = drm.unsupervised_outlier_finder_all(X)
        arr,drts,metrs = drm.result_array(res,y,'real')
        drama_all.append(arr)
        
        df = drm.sk_check(X,X,y,[1])
        for k,scr in enumerate(['AUC','MCC','RWS']):
            lof_all[k] = df[scr][0]
            ifr_all[k] = df[scr][1]
            
        drama_all = np.array(drama_all)

        np.save('../outputs/uns_drama_'+file_name+'_'+str(i),drama_all)
        np.save('../outputs/uns_lof_'+file_name+'_'+str(i),lof_all)
        np.save('../outputs/uns_ifr_'+file_name+'_'+str(i),ifr_all)
        
        cond = False
    except:
        pass





	

