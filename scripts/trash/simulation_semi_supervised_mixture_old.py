import matplotlib as mpl
mpl.use('agg')

import os
import sys
import numpy as np
import drama as drm
#import matplotlib.pylab as plt
#from matplotlib import gridspec

n_ftrs = 100 
noise = 0.8
scl = 0.00
sft = 0.00

i_sig = int(sys.argv[1])
n_train = int(sys.argv[2])
nn = int(sys.argv[3])
dir_add = './'+sys.argv[0][:-3]+'_res/'
drm.ch_mkdir(dir_add)

if os.path.exists(dir_add+str(i_sig)+'_'+str(n_train)+'_'+str(nn)+'.pickle'):
    exit()
    
x = np.linspace(0,1,n_ftrs)
X, y = drm.synt_mix(i_sig,n_ftrs,x=x,
                    n_inlier=1000,n_outlier=5,
                    sigma = noise,n1 = scl,n2 = sft,
                    n3 = scl,n4 = sft)
                    
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
    
y = (y!=i_sig).astype(int)
y = y[:,None]   

if n_train==0:
    res = drm.unsupervised_outlier_finder_all(X)
    df = drm.sk_check(X,X,y,[1])
    auc = []
    mcc = []
    rws = []
    for i in range(50):
        for j in ['real','latent']:
            o1 = res[j][i]
            auc.append(drm.roc_auc_score(y==1, o1))
            mcc.append(drm.MCC(y==1, o1))
            rws.append(drm.rws_score(y==1, o1))
            
#            print(y==1)
    auc = np.array(auc)
    mcc = np.array(mcc)
    rws = np.array(rws)
    drm.save(dir_add+str(i_sig)+'_'+str(n_train)+'_'+str(nn),[auc,mcc,rws,df])
    exit() 

iinds = np.argwhere(y[:,0]==0)[:,0]
oinds = np.argwhere(y[:,0]==1)[:,0]
nhalf = iinds.shape[0]//2

if oinds.shape[0]<=n_train:
    auc,mcc,rws,df = np.nan,np.nan,np.nan,np.nan
    drm.save(dir_add+file_names[ii]+'_'+str(n_train)+'_'+str(nn),[auc,mcc,rws,df])
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

auc_b = -100
mcc_b = -100
rws_b = -100

for i in range(50):
    for j in ['real']:
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

auc = drm.roc_auc_score(y_test==1, o1)
mcc = drm.MCC(y_test==1, o2)
rws = drm.rws_score(y_test==1, o3)

print(auc_set,mcc_set,rws_set)
print(auc,mcc,rws)

drm.save(dir_add+str(i_sig)+'_'+str(n_train)+'_'+str(nn),[auc,mcc,rws,df,[auc_set,mcc_set,rws_set]])



