
# coding: utf-8

# In[4]:


import drama as drm
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
from sklearn.metrics import roc_auc_score

import os
import glob
import h5py
import scipy.io as sio

get_ipython().magic(u'matplotlib inline')


# In[21]:


fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)[:10]
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]
print file_names


# In[22]:


frac = 0.3

drama_all = np.zeros((n_files,3))
lof_all = np.zeros((n_files,3))
ifr_all = np.zeros((n_files,3))

for i in range(len(fils)):
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
    for j,scr in enumerate(['AUC','MCC','RWS']):
        lof_all[i,j] = df[scr][0]
        ifr_all[i,j] = df[scr][1]


# In[24]:


iii = 0

crt = ['AUC','MCC','RWS']

for iii in range(3):
    ind = np.arange(n_files)  # the x locations for the groups
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(20,10))
    rects1 = ax.bar(ind - width, drama_all[:,iii], width, color='red', label='DRAMA')
    rects2 = ax.bar(ind , lof_all[:, iii], width, color='blue', label='LOF')
    rects3 = ax.bar(ind + width, ifr_all[:, iii], width, color='green', label='i-forest')

    ax.set_title('Averaged '+crt[iii],fontsize=25)
    ax.set_xticks(ind)
    ax.set_xticklabels(file_names,rotation=45,fontsize=15)
    ax.set_xlim(ind.min()-width,ind.max()+width)

    ax.set_ylabel(crt[iii],fontsize=20)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],fontsize=15)

    ax.set_xlim(-2*width,n_files-1+2*width)
    ax.legend(fontsize=15)

    # plt.savefig('recommendation_'+crt[iii]+'.jpg',dpi=150,bbox_inches='tight')

