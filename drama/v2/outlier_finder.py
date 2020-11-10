from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import pylab as plt
from pandas import DataFrame
from .splitters import Splitter
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import VALID_METRICS
from sklearn.ensemble import IsolationForest
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import braycurtis,canberra,chebyshev,cityblock
from scipy.spatial.distance import correlation,minkowski,wminkowski

from .utils import COLORS

all_metrics = ['cityblock','L2','L4','braycurtis',
               'canberra','chebyshev','correlation']

def plot_main_shapes(X,labels,cl=16):

    ll = int(np.sqrt(len(np.unique(labels))))
    l1 = ll+1
    l2 = len(np.unique(labels))//l1+1
    fig,axs = plt.subplots(l2,l1,figsize=(4*l1,4*l2))

    [axi.set_xticks([]) for axi in axs.ravel()]
    [axi.set_yticks([]) for axi in axs.ravel()]

    clrs = 4*COLORS
    for i in np.unique(labels).astype(int):
        X0 = X[labels==i]
        try:
            ax = axs[i//l1,i%l1]
        except:
            ax = axs[i//l1]
        ax.set_title(X0.shape[0],y=0.9)
        ax.plot(np.percentile(X0,50,axis=0),color=clrs[i])
        ax.plot(np.mean(X0,axis=0),ls='-.',color='k')
        ax.fill_between(np.arange(X0.shape[1]),
                         np.percentile(X0,cl,axis=0),
                         np.percentile(X0,100-cl,axis=0),
                         color=clrs[i],
                         alpha=0.5)

    plt.subplots_adjust(wspace=0.01,hspace=0.01)


def get_main_shapes(X,labels,trsh=0.1):
    main_shapes = []
    for i in np.unique(labels).astype(int):
        filt = labels==i
        if np.mean(filt)<trsh: continue
        X0 = X[filt]
        main_shapes.append(np.mean(X0,axis=0))
    return np.array(main_shapes)

def clip(x ,a_min=None, a_max=None):
    if a_min is None:
        a_min = x.min()
    if a_max is None:
        a_max = x.max()  
    return np.clip(x,a_min,a_max)

def L2(u, v):
    return minkowski(u, v, 2)
def L4(u, v):
    return minkowski(u, v, 4)
def expL4(u, v):
    x = clip(minkowski(u, v, 4),a_min=0,a_max=700)
    return np.exp(x)

dist_funcs = {'braycurtis':braycurtis,'canberra':canberra,'chebyshev':chebyshev,
                            'cityblock':cityblock,'correlation':correlation,'L2':L2
                            ,'L4':L4,'expL4':expL4}

def dist(metr,comp,X,w=None,w_norm=False):
    func = dist_funcs[metr]
    n_test = X.shape[0]
    n_c = comp.shape[0]
    all_dist = np.zeros(n_c)
    dist = np.zeros(n_test)
    for i in range(n_test):
        for j in range(n_c):
            if metr=='correlation':
                if np.std(X[i])==0:
                    X[i][0] += 1e-7
                if np.std(comp[j])==0:
                    comp[j][0] += 1e-7                    
            if w is None:
                all_dist[j] = func(X[i],comp[j])
            else:
                if w_norm:
                    w = w-w.min()+0.01
                    w = w/w.max()
                all_dist[j] = func(X[i],comp[j],w[j])
        dist[i] = np.nan_to_num(np.min(all_dist))
    return dist

def outliers_real(X,components,metrics=None):

    if isinstance(metrics, str):
        metrics = [metrics]

    if metrics is None:
        metrics = ['cityblock','L2','L4','expL4','braycurtis',
                   'canberra','chebyshev','correlation']
    distance = {}
    for metric in metrics:
        distance_test = dist(metric,np.array(components),X)
        distance[metric] = distance_test
    return distance

def MCC(outliers,v,n_o=None):
    if n_o is None:
        n_o = int(outliers.sum())
    o_ind = np.argsort(v)[::-1]
    o_ind = o_ind[:n_o]
    pred = np.zeros(outliers.shape)
    pred[o_ind] = 1
    return matthews_corrcoef(outliers.astype(int),pred)

def rws_score(outliers,v,n_o=None):
    if n_o is None:
        n_o = int(outliers.sum())
    b_s = np.arange(n_o)+1
    o_ind = np.argsort(v)[::-1]
    o_ind = o_ind[:n_o]
    return 1.*np.sum(b_s*outliers[o_ind].reshape(-1))/b_s.sum()

def golden_distance(X,labels,cl=16):
    distances = []
    for i in np.unique(labels).astype(int):
        X0 = X[labels==i]
        sm = np.percentile(X0,50,axis=0)
    #     np.mean(X0,axis=0)
        sl = np.percentile(X0,cl,axis=0)
        su = np.percentile(X0,100-cl,axis=0)
        distances.append(np.abs(X-sm)/(su-sl))
    distances = np.array(distances)
    return np.min(np.mean(distances,axis=-1),axis=0)
    

def sk_check(X_train,X_test,y_test,o_list):
    
    f_f = [LocalOutlierFactor(n_neighbors=5),\
    LocalOutlierFactor(n_neighbors=10),\
    LocalOutlierFactor(n_neighbors=35),\
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
    
def d_lof(X_seen,X_unseen=None,n_neighbors=20,algorithm='auto',metric='minkowski'):
    lof = LocalOutlierFactor(n_neighbors = n_neighbors,
                             algorithm = algorithm,
                             metric = metric,
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














