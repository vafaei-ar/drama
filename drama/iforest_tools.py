from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
from .utils import *
from sklearn.ensemble import IsolationForest

from sklearn.metrics import roc_auc_score

def d_iforest(X_seen,X_unseen=None,n_estimators=100,max_features=1.0,bootstrap=False):
    isof = IsolationForest(n_estimators=n_estimators,
                           max_features=max_features,
                           bootstrap=bootstrap,
                           behaviour="new",
                           n_jobs=-1)
    isof.fit(X_seen)
    if X_unseen is None:
        scores_pred = isof.decision_function(X_seen)
    else:
        scores_pred = isof.decision_function(X_unseen)
    return scores_pred.max()-scores_pred

def grid_run_iforest(X_seen,y_seen,
                     X_unseen=None,y_unseen=None,
                     n_estimators= [50,100,150],
                     max_features= [0.2,0.5,1.0],
                     bootstrap=[False,True]):
                 
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

    for ns in n_estimators:
        for mf in max_features:
            for bs in bootstrap:
                conf.append([ns,mf,bs])
                outliers = d_iforest(X_seen,X_unseen_p,n_estimators=ns,max_features=mf,bootstrap=bs)
                aucs.append(roc_auc_score(y_seen, outliers))
                mccs.append(MCC(y_seen, outliers))
                rwss.append(rws_score(y_seen, outliers))
                
    if semisupervised:
        ns,mf,bs = conf[np.argmax(aucs)]
        outliers = d_iforest(X_unseen,n_estimators=ns,max_features=mf,bootstrap=bs)
        auc = roc_auc_score(y_unseen, outliers)
        
        ns,mf,bs = conf[np.argmax(mccs)]
        outliers = d_iforest(X_unseen,n_estimators=ns,max_features=mf,bootstrap=bs)
        mcc = roc_auc_score(y_unseen, outliers)
        
        ns,mf,bs = conf[np.argmax(rwss)]
        outliers = d_iforest(X_unseen,n_estimators=ns,max_features=mf,bootstrap=bs)
        rws = roc_auc_score(y_unseen, outliers)
        
        return auc,mcc,rws
    
    else:
        return np.array(aucs),np.array(mccs),np.array(rwss),np.array(conf)




