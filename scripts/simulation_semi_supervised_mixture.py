import matplotlib as mpl
mpl.use('agg')

import os
import sys
import argparse
import numpy as np
import drama as drm
#import matplotlib.pylab as plt
#from matplotlib import gridspec
import warnings
warnings.filterwarnings("ignore", message='default contamination parameter 0.1 will change in version 0.22 to "auto". This will change the predict method behavior.')
warnings.filterwarnings("ignore", message='Data with input dtype float64 was converted to bool by check_pairwise_arrays.')
warnings.filterwarnings("ignore", message='Invalid value encountered in percentile')

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('--isig', action="store", type=int, required=True)
parser.add_argument('--ntrain', action="store", type=int, required=True)
parser.add_argument('--nn', action="store", type=int, required=True)
parser.add_argument('--nftrs', action="store", type=int, default=100)

args = parser.parse_args()
i_sig = args.isig
n_train = args.ntrain
nn = args.nn
n_ftrs = args.nftrs

dir_add = './'+sys.argv[0][:-3]+'_'+str(n_ftrs)+'_res/'
drm.ch_mkdir(dir_add)

noise = 0.8
scl = 0.00
sft = 0.00

i_sig = int(sys.argv[1])
n_train = int(sys.argv[2])
nn = int(sys.argv[3])
scn = int(sys.argv[4])

dir_add = './'+sys.argv[0][:-3]+'_'+str(scn)+'_'+str(n_ftrs)+'_res/'
drm.ch_mkdir(dir_add)

if os.path.exists(dir_add+str(i_sig)+'_'+str(n_train)+'_'+str(nn)+'.pickle'):
    exit()
    
if scn==1:
    ns = np.zeros(10)+5
    ns[i_sig-1] = 1000
elif scn==2:
    inds = np.arange(10)
    np.random.shuffle(inds)
    ns = np.zeros(10)+5
    for i in inds[:5]:
        ns[i] = 500
elif scn==3:    
    ns = np.zeros(10)+500
    ns[i_sig-1] = 50  
else:
    print('the scenario is not recognized!')
    exit()
    
numbers = {}
for i in range(10):
    numbers[i+1] = ns[i]

X,y = drm.simulate_shapes(numbers=numbers,n_ftrs = n_ftrs,
                            sigma=noise,
                            n1 = scl,n2 = sft,
					        n3 = scl,n4 = sft)

if scn==1:
    y = (y!=i_sig).astype(int)
elif scn==2:
    y = np.isin(y, inds[5:]+1).astype(int)
elif scn==3:    
    y = (y==i_sig).astype(int) 

#gs = gridspec.GridSpec(1, 2)
#plt.figure(figsize=(8,3)) 
#ax1 = plt.subplot(gs[0, 0])
#ax2 = plt.subplot(gs[0, 1])
#ax1.set_title('Inliers')
#ax2.set_title('Outliers')

#inliers = X[y==i_sig]
#outliers = X[y!=i_sig]
#outliers_y = y[y!=i_sig]

#for i in range(0,45,5):
#    ax1.plot(inliers[i],'b')
#    ax2.plot(outliers[i],drm.COLORS[outliers_y[i]])
#    
#plt.subplots_adjust(hspace=0.3,left=0.1, right=0.9, top=0.9, bottom=0.1)
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




