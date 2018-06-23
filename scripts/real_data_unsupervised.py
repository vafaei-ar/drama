
import drama as drm
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec

import os
import glob
import h5py
import scipy.io as sio

fils = sorted(glob.glob('../data/*.mat'), key=os.path.getsize)[:4]
n_files = len(fils)
file_names = [i.split('/')[-1][:-4] for i in fils]
print file_names

result = []
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
        
    res = drm.unsupervised_outlier_finder_all(X)
    arr,drts,metrs = drm.result_array(res,y,'real')
    result.append(arr)
    
    df = drm.sk_check(X,X,y,[1])
    for j,scr in enumerate(['AUC','MCC','RWS']):
        lof_all[i,j] = df[scr][0]
        ifr_all[i,j] = df[scr][1]
    
result = np.array(result)

drm.plot_table(np.mean(result,axis=0),drts,metrs,save=True)

auc = np.sum((result[:, :, :, 0].T>lof_all[:, 0]) & (result[:, :, :, 0].T>ifr_all[:, 0]),axis=-1).T
mcc = np.sum((result[:, :, :, 1].T>lof_all[:, 1]) & (result[:, :, :, 1].T>ifr_all[:, 1]),axis=-1).T
rws = np.sum((result[:, :, :, 2].T>lof_all[:, 2]) & (result[:, :, :, 2].T>ifr_all[:, 2]),axis=-1).T

fig = plt.figure(figsize=(20,10))
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect('auto')
res = ax.imshow(auc, cmap=plt.cm.jet,interpolation='nearest')

width, height = auc.shape

for x in xrange(width):
    for y in xrange(height):
        ax.annotate('AUC: {:d}\n MCC: {:d}\n RWS: {:d}'.format(auc[x][y],mcc[x][y],rws[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center',fontsize=18);

plt.xticks(range(10),['cityblock','L2','L4','braycurtis',
                         'canberra','chebyshev','correlation','mahalanobis','wL2','wL4'],fontsize=15)
plt.yticks(range(5), ['NMF','FastICA','PCA','AE','VAE'],fontsize=15)

plt.title('Number of successes (LOF and i-forest) out of 20 data set',fontsize=25)
plt.annotate('** Colors depend on AUC.', (0,0), (0, -30), xycoords='axes fraction',
             textcoords='offset points', va='top',fontsize=15)

plt.savefig('AND_success.jpg',dpi=150,bbox_inches='tight')

