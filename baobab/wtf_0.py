import matplotlib as mpl
mpl.use('agg')

import os
import sys 
sys.path.insert(0,'../drama/')
import drama as mce

import numpy as np
import pickle
import pandas as pd
from time import time
import glob
import h5py
import scipy.io as sio
from scipy.ndimage import imread

from sklearn import neighbors 
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA
from sklearn.metrics import roc_auc_score

agg = AgglomerativeClustering()

def data_provide(typ,smooth=False):
    X_train = np.load('OutlierData'+str(typ)+'/y_train.npy')
    y_train = np.load('OutlierData'+str(typ)+'/labels_train.npy')

    X_test = np.load('OutlierData'+str(typ)+'/y_test.npy')
    y_test = np.load('OutlierData'+str(typ)+'/labels_test.npy')
    
    if smooth:
        X_train = mce.smoother(X_train,5)
        X_test = mce.smoother(X_test,5)
    
    return X_train,y_train,X_test,y_test

def full_test(X_train,X_test,y_test,j,levs,o_list,z_dim=2):
    dim_rs ={'AE':'AE','VAE':'VAE','none':'none','PCA':PCA(n_components=z_dim),'NMF':NMF(n_components=2), 
             'FastICA':FastICA(n_components=2, max_iter=1000)}

    dr_name = ['AE','VAE','none','PCA','NMF','FastICA']
    dr_f = ['AE','VAE','none',PCA(n_components=z_dim),
            NMF(n_components=z_dim),FastICA(n_components=z_dim, max_iter=1000)]

    dim_rs ={dr_name[j]:dr_f[j]}
    # dim_rs ={'none':'none'}
    metrics = mce.metrics

    # metrics = ['mahalanobis']

    columns = ['DRT','level','Metric','detspace']+['AUC','MCC','BRU']
    n_row = 2*levs*len(dim_rs)*(len(metrics)+1)
    index = np.arange(n_row) # array of numbers for the number of samples
    df = pd.DataFrame(columns=columns, index = index)

    exec 'T_o ='+(' | '.join(['(y_test=='+str(i)+')' for i in o_list]))

    i = -1
    for dim_r, value in dim_rs.iteritems():

        print '---------------  '+dim_r+'  ----------------'
        t0 = time()
        for lev in range(levs):
            # Splitter definition
            splitter = mce.Splitter(X_train, reducer = value, clustering = agg.fit_predict, z_dim=z_dim)

            # Splitting
            splitter.split(1,verbose=0,training_epochs=20)

            # outlier extraction for all of requeste metrics
            outliers_r = mce.outliers(X_test,splitter,metrics)
            outliers_l = mce.outliers_latent(splitter,metrics) 

            for metr in metrics:

                i += 1
                df['DRT'][i] = dim_r
                df['level'][i] = lev+1
                df['Metric'][i] = metr
                df['detspace'][i] = 'real'
                df['AUC'][i] = roc_auc_score(T_o, outliers_r[metr])
                df['MCC'][i] = mce.MCC(T_o, outliers_r[metr])
                df['BRU'][i] = mce.bru_score(T_o, outliers_r[metr])

                i += 1
                df['DRT'][i] = dim_r
                df['level'][i] = lev+1
                df['Metric'][i] = metr
                df['detspace'][i] = 'latent'                  
                df['AUC'][i] = roc_auc_score(T_o, outliers_l[metr])
                df['MCC'][i] = mce.MCC(T_o, outliers_l[metr])
                df['BRU'][i] = mce.bru_score(T_o, outliers_l[metr])

            i += 1
            df['DRT'][i] = dim_r
            df['level'][i] = lev+1
            df['Metric'][i] = 'ens'
            df['detspace'][i] = 'real'  
            ens_out = mce.norm_ensemble(outliers_r,0.1)
            df['AUC'][i] = roc_auc_score(T_o, ens_out)
            df['MCC'][i] = mce.MCC(T_o, ens_out)
            df['BRU'][i] = mce.bru_score(T_o, ens_out)

            i += 1
            df['DRT'][i] = dim_r
            df['level'][i] = lev+1
            df['Metric'][i] = 'ens'
            df['detspace'][i] = 'latent' 
            ens_out = mce.norm_ensemble(outliers_l,0.1)
            df['AUC'][i] = roc_auc_score(T_o, ens_out)
            df['MCC'][i] = mce.MCC(T_o, ens_out)
            df['BRU'][i] = mce.bru_score(T_o, ens_out)        
        t1 = time()
        
    return df,t1-t0

def sk_check(X_train,X_test,y_test,o_list):
    f_f = [neighbors.LocalOutlierFactor(n_neighbors=5),\
    neighbors.LocalOutlierFactor(n_neighbors=10),\
    neighbors.LocalOutlierFactor(n_neighbors=35),\
    IsolationForest(max_samples='auto')]
    f_name = ['LOF5','LOF10','LOF35','i-forest']

    columns = ['method']+['AUC','MCC','BRU']
    n_row = 2
    index = np.arange(n_row) # array of numbers for the number of samples
    df = pd.DataFrame(columns=columns, index = index)

    exec 'T_o ='+(' | '.join(['(y_test=='+str(i)+')' for i in o_list]))

    auc_max = -1
    for i in range(3):
        lof = f_f[i]
        lof.fit(X_test)
        outliers = -lof.negative_outlier_factor_

        auc_test = roc_auc_score(T_o, outliers)
        if auc_test>auc_max:
            auc_max = auc_test
            df['method'][0] = f_name[i]
            df['MCC'][0] = mce.MCC(T_o, outliers)
            df['AUC'][0] = auc_max
            df['BRU'][0] = mce.bru_score(T_o, outliers)

    df['method'][1] = f_name[3]
    isof = f_f[3]
    isof.fit(X_train)
    scores_pred = isof.decision_function(X_test)
    outliers = scores_pred.max()-scores_pred
    df['MCC'][1] = mce.MCC(T_o, outliers)
    df['AUC'][1] = roc_auc_score(T_o, outliers)
    df['BRU'][1] = mce.bru_score(T_o, outliers)

    return df

dr_name = ['AE','VAE','none','PCA','NMF','FastICA']

levs = 3
o_list = [1]
z_dim = 2

j = int(sys.argv[1])
i = int(sys.argv[2])

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
        xp = np.array(xp,dtype=int)
        x[i] = xp
        if 'B' in add:
            y[i] = 0
        else:
            y[i] = 1  

    return x,y

X_test, y_test = batch()

add = './res/'
mce.ch_mkdir(add)

if j<6:
	res = full_test(X_test,X_test,y_test,j,levs,o_list)
	with open(add+dr_name[j]+'_'+str(i)+'.pkl', 'wb') as f:
		  pickle.dump(res, f)

else:
	res = sk_check(X_test,X_test,y_test,o_list)
	with open(add+'sk_'+str(i)+'.pkl', 'wb') as f:
		  pickle.dump(res, f)







	

