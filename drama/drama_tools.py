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

import warnings
warnings.filterwarnings("ignore", message='default contamination parameter 0.1 will change in version 0.22 to "auto". This will change the predict method behavior.')
warnings.filterwarnings("ignore", message='Data with input dtype float64 was converted to bool by check_pairwise_arrays.')
warnings.filterwarnings("ignore", message='Invalid value encountered in percentile')


DEBUG = True
all_metrics = ['cityblock','L2','L4','braycurtis',
           'canberra','chebyshev','correlation','mahalanobis',
           'wL2','wL4']

def outliers_real(X,splitter,metrics):

    if splitter.convolutional:
        X = X.reshape(-1,splitter.n_ftrs)
        
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

class D_Drama(object):

    def __init__(self,X_seen,
                 drt_name='FastICA',
                 clustering=None,
                 z_dim=2):
                 
        dim_rs ={'DAE1D':'DAE1D','DVAE1D':'DVAE1D','CAE1D':'CAE1D','CVAE1D':'CVAE1D', 
                 'CAE2D':'CAE2D','CVAE2D':'CVAE2D', 
                 'PCA':PCA(n_components=z_dim),'NMF':NMF(n_components=z_dim), 
                 'FastICA':FastICA(n_components=z_dim, max_iter=1000)}

        if drt_name not in dim_rs.keys():   		
            print('Selected dimensionality reduction name is not recognized \n'+\
                  'Please choose one from:',dim_rs.keys())
            return
                    
        if clustering is None:
            agg = AgglomerativeClustering()
            clustering = agg.fit_predict
            
        data_dim = 1
        if X_seen.ndim==4:
            data_dim = 2
            
        self.splitter = Splitter(X_seen, reducer = dim_rs[drt_name], clustering = clustering, data_dim = data_dim, z_dim=z_dim)
        self.X_seen = X_seen

    def split(self,n_split=1):  
        self.splitter.split(n_split,verbose=0,training_epochs=20)
        
    def __call__(self,X_unseen=None,
                 metrics=all_metrics,
                 n_split=1,space='real'):  
                            
        if X_unseen is None:
            X_unseen = self.X_seen
            if n_split>1:
                print('Warning! too large n_split in anomaly detection may decrease performance.')
        
        outliers = {'real':None,'latent':None}

        # Splitting
        self.splitter.split(n_split,verbose=0,training_epochs=20)

        # outlier extraction for all of requeste metrics
        if space=='real':
            outliers['real'] = outliers_real(X_unseen,self.splitter,metrics)
        elif space=='latent':
            assert 0,'Latent space is under construction!'
    #        outliers['latent'] = outliers_latent(splitter,metrics)
        else:
            assert 0,'Space is not recognized!'
    #        outliers['real'] = outliers_real(X_unseen,splitter,metrics)
    #        outliers['latent'] = outliers_latent(splitter,metrics)
            
        return outliers

#'DAE1D','DVAE1D','CAE1D','CVAE1D','CAE2D','CVAE2D'

def grid_run_drama(X_seen,y_seen,
                   X_unseen=None,y_unseen=None,
                   drt_list = ['DAE1D','DVAE1D','PCA','NMF','FastICA'],
                   metrics = all_metrics,
                   n_split = 1):
                   
    '''         
    This function is able to deal with three modes:
    1- Unsupervised outlier detection 
    2- Semi-supervised outlier detection
    3- Novelty detection  
    '''      
    
    novelty = 0   
    semisupervised = 0 
    if (np.all(y_seen==0)) | (y_seen is None):
        novelty = 1
        X_unseen_p = X_unseen
        y_seen = y_unseen
        print('Novelty detection mode.')
        conds = (X_unseen is not None and y_unseen is not None)
        assert conds,'In novelty detection you need to input the unseen data sets.'
    elif y_unseen is not None and X_unseen is not None:
        semisupervised = 1
#        print('Semi-supervised option is not available for novelty detection.')
        X_unseen_p = None
        print('Semi-supervised outlier detection mode.')
    elif X_seen is not None:
        X_unseen_p = X_unseen
        print('Unsupervised outlier detection mode.')
    else:
        assert 0, 'The configuration is not recognized!'

    aucs,mccs,rwss,conf = [],[],[],[]

    X_seen = X_seen/X_seen.max()
    if not X_unseen is None:
        X_unseen = X_unseen/X_unseen.max()
        
    for drt in drt_list:
        d_drama = D_Drama(X_seen = X_seen,drt_name = drt)
        for nsp in range(n_split):
            outliers_rep = d_drama(X_unseen=X_unseen_p, n_split = 1)
            for metr in metrics:
                o1 = outliers_rep['real'][metr]
    #            o2 = outliers_rep['latent'][metr]
                
                auc = roc_auc_score(y_seen==1, o1)
                mcc = MCC(y_seen==1, o1)
                rws = rws_score(y_seen==1, o1)
                
                aucs.append(auc)
                mccs.append(mcc)
                rwss.append(rws)
                conf.append(['real',drt,metr,nsp+1])
            
    if semisupervised:
        space,drt,metric,nsp = conf[np.argmax(aucs)]
        d_drama = D_Drama(X_seen = X_unseen, drt_name = drt)
        res = d_drama(metrics=metric,n_split=nsp)
        outliers = res[space][metric]    
        auc = roc_auc_score(y_unseen, outliers)
        
        space,drt,metric,nsp = conf[np.argmax(mccs)]
        d_drama = D_Drama(X_seen = X_unseen, drt_name = drt)
        res = d_drama(metrics=metric,n_split=nsp)
        outliers = res[space][metric]  
        mcc = roc_auc_score(y_unseen, outliers)
        
        space,drt,metric,nsp = conf[np.argmax(rwss)]
        d_drama = D_Drama(X_seen = X_unseen, drt_name = drt)
        res = d_drama(metrics=metric,n_split=nsp)
        outliers = res[space][metric]  
        rws = roc_auc_score(y_unseen, outliers)
        
        return auc,mcc,rws
    
    else:
        return np.array(aucs),np.array(mccs),np.array(rwss),np.array(conf)
    

#def grid_run_novelty_drama(X_train,X,n_split=2):

#    X_train = X_train/X_train.max()
#    X = X/X.max()

#    res = {'drt':[],'metric':[],'pr':[],'real':[],'latent':[]}
#    
#    drt_list = ['AE','VAE','PCA','NMF','FastICA']

#    num = X.shape[0]
#    n_out = num//20

#    for drt in drt_list:

#        outliers_rep = get_novelties(X_train,X,drt,all_metrics,n_split=n_split)

#        for metr in all_metrics:
#            o1 = outliers_rep['real'][metr]
#            o2 = outliers_rep['latent'][metr]

##            pr1 = pearsonr(o1[-n_out:],o2[-n_out:])[0]
#            pr = pearsonr(np.argsort(o1)[-n_out:],np.argsort(o2)[-n_out:])[0]

#            res['drt'].append(drt)
#            res['metric'].append(metr)
#            res['pr'].append(pr)
#            res['real'].append(o1)
#            res['latent'].append(o2)
#            
#    for i in ['pr','real','latent']:        
#        res[i] = np.array(res[i])

#    return res


