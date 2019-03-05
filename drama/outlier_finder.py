from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from .utils import *
from .splitters import Splitter
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from pandas import DataFrame
from sklearn import neighbors 
from sklearn.ensemble import IsolationForest

DEBUG = True
all_metrics = ['cityblock','L2','L4','braycurtis',
           'canberra','chebyshev','correlation','mahalanobis',
           'wL2','wL4']

def sk_check(X_train,X_test,y_test,o_list):
    
    f_f = [neighbors.LocalOutlierFactor(n_neighbors=5),\
    neighbors.LocalOutlierFactor(n_neighbors=10),\
    neighbors.LocalOutlierFactor(n_neighbors=35),\
    IsolationForest(max_samples='auto')]
    f_name = ['LOF5','LOF10','LOF35','i-forest']

    columns = ['method']+['AUC','MCC','RWS']
    n_row = 2
    index = np.arange(n_row) # array of numbers for the number of samples
    df = DataFrame(columns=columns, index = index)
    y_test = np.array(y_test)
    exec('T_o ='+(' | '.join(['(y_test=='+str(i)+')' for i in o_list])),locals(),globals())

    auc_max = -1
    for i in range(3):
        lof = f_f[i]
        lof.fit(X_test)
        outliers = -lof.negative_outlier_factor_

        auc_test = roc_auc_score(T_o, outliers)
        if auc_test>auc_max:
            auc_max = auc_test
            df['method'][0] = f_name[i]
            df['MCC'][0] = MCC(T_o, outliers)
            df['AUC'][0] = auc_max
            df['RWS'][0] = rws_score(T_o, outliers)

    df['method'][1] = f_name[3]
    isof = f_f[3]
    isof.fit(X_train)
    scores_pred = isof.decision_function(X_test)
    outliers = scores_pred.max()-scores_pred
    df['MCC'][1] = MCC(T_o, outliers)
    df['AUC'][1] = roc_auc_score(T_o, outliers)
    df['RWS'][1] = rws_score(T_o, outliers)

    return df

def ind2score(oi):
    num = oi.shape[0]
    score = np.zeros(num)
    score[oi[::-1]] = np.linspace(0,1,num)
    return score

def outliers_real(X,splitter,metrics):

    if isinstance(metrics, str):
        metrics = [metrics]

    nmetrics = ['cityblock','L2','L4','expL4','braycurtis',
             'canberra','chebyshev','correlation']
    wmetrics = ['wL2','wL4','wexpL4']

    n_data = X.shape[0]
    n_ftrs = X.shape[1]
    assert splitter.n_ftrs==n_ftrs,'Inconsistence dimension! '+str(splitter.n_ftrs)+'!='+str(n_ftrs)
    distance = {}

    mean_vec = splitter.mean_vector[-1].reshape(-1,n_ftrs)
    cov = splitter.covariance[-1].reshape(-1,n_ftrs,n_ftrs)
    w = np.array([1./np.sqrt(clip(np.diag(cv),a_min=1e-4)) for cv in cov])
    components = splitter.mean_components[-1].reshape(-1,n_ftrs)

    if 'mahalanobis' in metrics:
        nc = cov.shape[0]
        cov_inv = np.zeros(cov.shape)
        for i in range(nc):
          cov_inv[i] = np.linalg.pinv(cov[i])

        distance_test = np.zeros(n_data)
        for i in range(n_data):
          dst_arr = np.zeros(nc)
          for j in range(nc):
            vec = (X[i]-mean_vec[j]).reshape(n_ftrs,1)
            dst_arr[j] = np.matmul(np.matmul(vec.T,cov_inv[j]),vec)

          distance_test[i] = np.min(dst_arr)

#        distance_test[i] = np.min(dst_arr)
        distance['mahalanobis'] = distance_test
              
    for metric in metrics:
        if metric in nmetrics:
            distance_test = dist(metric,np.array(components),X)
            distance[metric] = distance_test
          
            if np.any(np.isnan(distance_test)) and DEBUG:
                print('There is a problem with '+metric)
                return distance_test
          
        elif metric in wmetrics:
            distance_test = dist(metric,np.array(components),X,w)
            distance[metric] = distance_test
            
            if np.any(np.isnan(distance_test)) and DEBUG:
                print('There is a problem with '+metric)
                return distance_test

    return distance

def norm_ensemble(outliers,alpha=0.1):

    assert isinstance(outliers, dict),'Input should be a dictionary contains outliers using a several metrics.'      
    x = dic2array(outliers)
    x = x.view((float, len(x.dtype.names)))
    x = x-np.min(x,axis=0,keepdims=True)
    x_max = np.max(x,axis=0,keepdims=True)
    x = np.where(x_max!=0,x/x_max,0)
    return np.sum(np.power(x,alpha),axis=1)

def max_ensemble(outliers):

    assert isinstance(outliers, dict),'Input should be a dictionary contains outliers using a several metrics.'      
    x = dic2array(outliers)
    x = x.view((float, len(x.dtype.names)))
    x = x-np.min(x,axis=0,keepdims=True)
    x_max = np.max(x,axis=0,keepdims=True)
    x = np.where(x_max!=0,x/x_max,0)
    return np.max(x,axis=1)

def outliers_latent(splitter,metrics):

    if isinstance(metrics, str):
        metrics = [metrics]

    nmetrics = ['cityblock','L2','L4','expL4','braycurtis',
             'canberra','chebyshev','correlation']
    wmetrics = ['wL2','wL4','wexpL4']

    distance = {}
    n_c = 2*len(splitter.z_mu)

    Z = np.zeros((0,splitter.z_dim))
    for i in splitter.z_mu:
        if i.shape[0]!=0:
            Z = np.concatenate((Z,i),axis=0)

    z_mu = [Z[splitter.labels[-1]==i] for i in range(n_c)]

    mean_vec = np.zeros((n_c,splitter.z_dim))
    cov = np.zeros((n_c,splitter.z_dim,splitter.z_dim))

    for i in range(n_c):
        cov[i] = Cov_mat(z_mu[i])
        if z_mu[i].shape[0]==0:
            continue
        mean_vec[i] = np.mean(z_mu[i],axis=0)

    w = np.array([1./np.sqrt(clip(np.diag(cv),a_min=1e-4)) for cv in cov])
    components = np.zeros((0,splitter.z_dim))
    for i in splitter.mean_points[-1]:
          components = np.concatenate((components,i),axis=0)

    if 'mahalanobis' in metrics:
        nc = cov.shape[0]
        cov_inv = np.zeros(cov.shape)
        for i in range(nc):
          cov_inv[i] = np.linalg.pinv(cov[i])

        distance_test = np.zeros(splitter.n_samples)
        for i in range(splitter.n_samples):
          dst_arr = np.zeros(nc)
          for j in range(nc):
            vec = (Z[i]-mean_vec[j]).reshape(splitter.z_dim,1)
            dst_arr[j] = np.matmul(np.matmul(vec.T,cov_inv[j]),vec)

          distance_test[i] = np.min(dst_arr)

#        distance_test[i] = np.min(dst_arr)
        distance['mahalanobis'] = distance_test

    for metric in metrics:
        if metric in nmetrics:
            distance_test = dist(metric,np.array(components),Z)           
            distance[metric] = distance_test 
                       
            if np.any(np.isnan(distance_test)) and DEBUG:
                print('There is a problem with '+metric)
                return distance_test

        elif metric in wmetrics:
            distance_test = dist(metric,np.array(components),Z,w)
            distance[metric] = distance_test
            
            if np.any(np.isnan(distance_test)) and DEBUG:
                print('There is a problem with '+metric)
                return distance_test

    return distance
    
def get_outliers(X,drt_name,metrics,clustering=None,z_dim=2,space='both'):

    dim_rs ={'AE':'AE','VAE':'VAE','PCA':PCA(n_components=z_dim),'NMF':NMF(n_components=z_dim), 
             'FastICA':FastICA(n_components=z_dim, max_iter=1000)}
             
    if drt_name not in dim_rs.keys():   		
        print('Selected dimensionality reduction name is not recognized \n'+\
              'Please choose one from:',dim_rs.keys())
        return
        
    outliers = {'real':None,'latent':None}

    if clustering is None:
        agg = AgglomerativeClustering()
        clustering = agg.fit_predict
        
    splitter = Splitter(X, reducer = dim_rs[drt_name], clustering = clustering, z_dim=z_dim)

    # Splitting
    splitter.split(1,verbose=0,training_epochs=20)

    # outlier extraction for all of requeste metrics
    if space=='real':
        outliers['real'] = outliers_real(X,splitter,metrics)
    elif space=='latent':
        outliers['latent'] = outliers_latent(splitter,metrics)
    else:
        outliers['real'] = outliers_real(X,splitter,metrics)
        outliers['latent'] = outliers_latent(splitter,metrics)
        
    return outliers
    
def get_novelties(X_train,X,drt_name,metrics,clustering=None,n_slpit=2,z_dim=2,space='both'):

    dim_rs ={'AE':'AE','VAE':'VAE','PCA':PCA(n_components=z_dim),'NMF':NMF(n_components=z_dim), 
             'FastICA':FastICA(n_components=z_dim, max_iter=1000)}
             
    if drt_name not in dim_rs.keys():   		
        print('Selected dimensionality reduction name is not recognized \n'+\
              'Please chose one from:',dim_rs.keys())
        return
        
    outliers = {'real':None,'latent':None}

    if clustering is None:
        agg = AgglomerativeClustering()
        clustering = agg.fit_predict
        
    splitter = Splitter(X_train, reducer = dim_rs[drt_name], clustering = clustering, z_dim=z_dim)

    # Splitting
    splitter.split(n_slpit,verbose=0,training_epochs=20)

    # outlier extraction for all of requeste metrics
    if space=='real':
        outliers['real'] = outliers_real(X,splitter,metrics)
    elif space=='latent':
        outliers['latent'] = outliers_latent(splitter,metrics)
    else:
        outliers['real'] = outliers_real(X,splitter,metrics)
        outliers['latent'] = outliers_latent(splitter,metrics)
        
    return outliers

def unsupervised_outlier_finder_all(X):

    X = X/X.max()

    res = {'drt':[],'metric':[],'pr':[],'real':[],'latent':[]}
    
    drt_list = ['AE','VAE','PCA','NMF','FastICA']

    num = X.shape[0]
    n_out = num//20

    for drt in drt_list:

        outliers_rep = get_outliers(X,drt,all_metrics)

        for metr in all_metrics:
            o1 = outliers_rep['real'][metr]
            o2 = outliers_rep['latent'][metr]

#            pr1 = pearsonr(o1[-n_out:],o2[-n_out:])[0]
            pr = pearsonr(np.argsort(o1)[-n_out:],np.argsort(o2)[-n_out:])[0]

            res['drt'].append(drt)
            res['metric'].append(metr)
            res['pr'].append(pr)
            res['real'].append(o1)
            res['latent'].append(o2)
            
    for i in ['pr','real','latent']:        
        res[i] = np.array(res[i])

    return res
    
def result_array(res,y,space):
    n_drt = len(np.unique(res['drt']))
    n_metr = len(np.unique(res['metric']))

    arr = np.zeros((n_drt,n_metr,3))
    drts = n_drt*['']
    metrs = n_metr*['']

    for i in range(n_drt*n_metr):
        row = i // n_metr
        col = i % n_metr
        o1 = res[space][i]

        auc = roc_auc_score(y==1, o1)
        mcc = MCC(y==1, o1)
        bru = rws_score(y==1, o1)

        arr[row,col,0] = auc
        arr[row,col,1] = mcc
        arr[row,col,2] = bru
        drts[row] = res['drt'][i]
        metrs[col] = res['metric'][i]
        
    return arr,drts,metrs

def supervised_outlier_finder_all(X_train,y_train,X_test):    
    X_train = X_train/X_train.max()
    X_test = X_test/X_test.max()
    
    res = unsupervised_outlier_finder_all(X_train)        
        
    auc = []
    mcc = []
    rws = []
    
    auc_b = -100
    mcc_b = -100
    rws_b = -100
    
    for i in range(50):
        for j in ['real','latent']:
            o1 = res[j][i]
            auc = roc_auc_score(y_train==1, o1)
            mcc = MCC(y_train==1, o1)
            rws = rws_score(y_train==1, o1)
            
            if auc_b<auc:
                auc_b = auc
                auc_set = [j,res['drt'][i],res['metric'][i]]

            if mcc_b<mcc:
                mcc_b = mcc
                mcc_set = [j,res['drt'][i],res['metric'][i]]

            if rws_b<rws:
                rws_b = rws
                rws_set = [j,res['drt'][i],res['metric'][i]]
                
    
    res = get_outliers(X_test,auc_set[1],auc_set[2],clustering=None,z_dim=2,space=auc_set[0])
    o1 = res[auc_set[0]][auc_set[2]]

    res = get_outliers(X_test,mcc_set[1],mcc_set[2],clustering=None,z_dim=2,space=mcc_set[0])
    o2 = res[mcc_set[0]][mcc_set[2]]
    
    res = get_outliers(X_test,rws_set[1],rws_set[2],clustering=None,z_dim=2,space=rws_set[0])
    o3 = res[rws_set[0]][rws_set[2]]
    
    return o1,o2,o3

def novelty_finder_all(X_train,X,n_slpit=2):

    X_train = X_train/X_train.max()
    X = X/X.max()

    res = {'drt':[],'metric':[],'pr':[],'real':[],'latent':[]}
    
    drt_list = ['AE','VAE','PCA','NMF','FastICA']

    num = X.shape[0]
    n_out = num//20

    for drt in drt_list:

        outliers_rep = get_novelties(X_train,X,drt,all_metrics,n_slpit=n_slpit)

        for metr in all_metrics:
            o1 = outliers_rep['real'][metr]
            o2 = outliers_rep['latent'][metr]

#            pr1 = pearsonr(o1[-n_out:],o2[-n_out:])[0]
            pr = pearsonr(np.argsort(o1)[-n_out:],np.argsort(o2)[-n_out:])[0]

            res['drt'].append(drt)
            res['metric'].append(metr)
            res['pr'].append(pr)
            res['real'].append(o1)
            res['latent'].append(o2)
            
    for i in ['pr','real','latent']:        
        res[i] = np.array(res[i])

    return res

def plot_table(arr,drts,metrs,save=False,prefix=''):
    import matplotlib.pylab as plt
    
    crt = ['AUC','MCC','RWS']
    n_drt = len(drts)
    n_metr = len(metrs)
    for iii in range(3):
        mtx = arr[:,:,iii]

        fig = plt.figure(figsize=(2*n_metr,2*n_drt))
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect('auto')
        ax.imshow(mtx, cmap=plt.cm.jet,interpolation='nearest')

        width, height = mtx.shape

        rnk = 50-mtx.ravel().argsort().argsort().reshape(mtx.shape)

        for x in range(width):
            for y in range(height):
                ax.annotate('{:3.1f}\n rank: {:d}'.format(100*mtx[x][y],rnk[x][y]), xy=(y, x), 
                            horizontalalignment='center',
                            verticalalignment='center',fontsize=20);

        plt.xticks(np.arange(n_metr),metrs,fontsize=20,rotation=20)
        plt.yticks(np.arange(n_drt), drts,fontsize=20)

        plt.title(crt[iii]+r' ($\%$)',fontsize=25)
        if save:
            plt.savefig(prefix+crt[iii]+'.jpg',dpi=150,bbox_inches='tight')
