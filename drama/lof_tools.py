from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from .utils import *
from warnings import warn
from sklearn.metrics import roc_auc_score

#from pandas import DataFrame
from sklearn.neighbors import LocalOutlierFactor
#from sklearn.ensemble import IsolationForest
from sklearn.neighbors import VALID_METRICS

def d_lof(X_seen,X_unseen=None,n_neighbors=20,algorithm='auto',metric='minkowski'):
    lof = LocalOutlierFactor(n_neighbors = n_neighbors,
                             algorithm = algorithm,
                             metric = metric,
                             contamination='auto',
                             novelty=not (X_unseen is None),
                             n_jobs=-1)
    lof.fit(X_seen)
    if X_unseen is None:
        return -lof.negative_outlier_factor_
    else:
        return -lof.score_samples(X_unseen)

def grid_run_lof(X_seen,y_seen=None,
                 X_unseen=None,y_unseen=None,
                 n_neighbors = [5,20,35],
                 algorithms = ['ball_tree', 'kd_tree', 'brute'],
                 metrics=None):
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

    for nn in n_neighbors:
        for al in algorithms:
            if metrics is None:
                metrics = VALID_METRICS[al]
            for mt in metrics:
                try:
                    outliers = d_lof(X_seen=X_seen,X_unseen=X_unseen_p,n_neighbors=nn,algorithm=al,metric=mt)
                    conf.append([nn,al,mt])
                    aucs.append(roc_auc_score(y_seen, outliers))
                    mccs.append(MCC(y_seen, outliers))
                    rwss.append(rws_score(y_seen, outliers))
                except:
                    pass

                    
    if semisupervised:
        nn,al,mt = conf[np.argmax(aucs)]
        outliers = d_lof(X_seen=X_unseen,n_neighbors=nn,algorithm=al,metric=mt)
        auc = roc_auc_score(y_unseen, outliers)
        
        nn,al,mt = conf[np.argmax(mccs)]
        outliers = d_lof(X_seen=X_unseen,n_neighbors=nn,algorithm=al,metric=mt)
        mcc = roc_auc_score(y_unseen, outliers)
        
        nn,al,mt = conf[np.argmax(rwss)]
        outliers = d_lof(X_seen=X_unseen,n_neighbors=nn,algorithm=al,metric=mt)
        rws = roc_auc_score(y_unseen, outliers)
        
        return auc,mcc,rws
    
    else:
        return np.array(aucs),np.array(mccs),np.array(rwss),np.array(conf)

#def sk_check(X_train,X_test,y_test,o_list):
#    
#    f_f = [LocalOutlierFactor(n_neighbors=5),\
#    LocalOutlierFactor(n_neighbors=10),\
#    LocalOutlierFactor(n_neighbors=35),\
#    IsolationForest(max_samples='auto')]
#    f_name = ['LOF5','LOF10','LOF35','i-forest']

#    columns = ['method']+['AUC','MCC','RWS']
#    n_row = 2
#    index = np.arange(n_row) # array of numbers for the number of samples
#    df = DataFrame(columns=columns, index = index)
#    y_test = np.array(y_test)
#    exec('T_o ='+(' | '.join(['(y_test=='+str(i)+')' for i in o_list])),locals(),globals())

#    auc_max = -1
#    for i in range(3):
#        lof = f_f[i]
#        lof.fit(X_test)
#        outliers = -lof.negative_outlier_factor_

#        auc_test = roc_auc_score(T_o, outliers)
#        if auc_test>auc_max:
#            auc_max = auc_test
#            df['method'][0] = f_name[i]
#            df['MCC'][0] = MCC(T_o, outliers)
#            df['AUC'][0] = auc_max
#            df['RWS'][0] = rws_score(T_o, outliers)

#    df['method'][1] = f_name[3]
#    isof = f_f[3]
#    isof.fit(X_train)
#    scores_pred = isof.decision_function(X_test)
#    outliers = scores_pred.max()-scores_pred
#    df['MCC'][1] = MCC(T_o, outliers)
#    df['AUC'][1] = roc_auc_score(T_o, outliers)
#    df['RWS'][1] = rws_score(T_o, outliers)

#    return df
