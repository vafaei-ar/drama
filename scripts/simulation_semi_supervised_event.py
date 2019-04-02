import matplotlib as mpl
mpl.use('agg')

import os
import sys
import numpy as np
import drama as drm
#import matplotlib.pylab as plt
#from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore", message='default contamination parameter 0.1 will change in version 0.22 to "auto". This will change the predict method behavior.')
warnings.filterwarnings("ignore", message='Data with input dtype float64 was converted to bool by check_pairwise_arrays.')
warnings.filterwarnings("ignore", message='Invalid value encountered in percentile')

n_ftrs = 100 
noise = 0.3
scl = 0.00
sft = 0.00

i_sig = int(sys.argv[1])
n_train = int(sys.argv[2])
nn = int(sys.argv[3])
dir_add = './'+sys.argv[0][:-3]+'_'+str(n_ftrs)+'_res/'
drm.ch_mkdir(dir_add)

if os.path.exists(dir_add+str(i_sig)+'_'+str(n_train)+'_'+str(nn)+'.pickle'):
    exit()
    
X, y = drm.synt_event(i_sig,n_ftrs,
                      n_inlier=1000,n_outlier=50,
                      sigma = noise,n1 = scl,
                      n2 = sft,n3 = scl,n4 = sft)   
                                      
#gs = gridspec.GridSpec(1, 2)
#plt.figure(figsize=(8,3)) 
#ax1 = plt.subplot(gs[0, 0])
#ax2 = plt.subplot(gs[0, 1])
#ax1.set_title('Inliers')
#ax2.set_title('Outliers')

#inliers = X[y==0]
#outliers = X[y==1]
#for i in range(10):
#    ax1.plot(inliers[i],'b')
#    ax2.plot(outliers[i],'r') 
#plt.savefig('1.jpg')
y = y[:,None] 

if n_train==0:
    dd = drm.grid_run_drama(X_seen=X,y_seen=y)
    ll = drm.grid_run_lof(X_seen=X,y_seen=y)
    ii = drm.grid_run_iforest(X_seen=X,y_seen=y)
    
else:
    X_train,y_train,X_test,y_test = drm.data_split(X,y,n_train)
    dd = drm.grid_run_drama(X_seen=X_train ,y_seen=y_train ,X_unseen=X_test, y_unseen=y_test, n_split=1)
    ll = drm.grid_run_lof(X_seen=X_train ,y_seen=y_train ,X_unseen=X_test, y_unseen=y_test)
    ii = drm.grid_run_iforest(X_seen=X_train ,y_seen=y_train ,X_unseen=X_test, y_unseen=y_test)

drm.save(dir_add+str(i_sig)+'_'+str(n_train)+'_'+str(nn),[dd,ll,ii])















