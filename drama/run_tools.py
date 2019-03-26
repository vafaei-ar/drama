from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .signal_synthesis import *
import numpy as np

def synt_event(i_sig, n_ftrs,x=None,n_inlier=2000,n_outlier=100,
								sigma = 0.2,n1 = 0.02,n2 = 0.01,n3 = 0.02,n4 = 0.01,
								mu=[0,1],amp=[0.3,0.4],sig=[0.08,0.1]):

    main_data = {i_sig:n_inlier}
    event_data = {i_sig:n_outlier}
    if x is None:
        x = np.linspace(0,1,n_ftrs)

    X = []
    y = []
    for key,value in main_data.items():
        for _ in range(value):
            Xp = signal(key,x,sigma,n1,n2,n3,n4)
            X.append(Xp)
            y.append(0)

    for key,value in event_data.items():
        for _ in range(value):
            Xp = signal(key,x,sigma,n1,n2,n3,n4)
            Xp = event_sig(Xp,mu=mu,amp=amp,sig=sig)
            X.append(Xp)
            y.append(1)
            
    return np.array(X),np.array(y)

def synt_mix(i_sig, n_ftrs,x=None,n_sig=11,n_inlier=1000,n_outlier=5,sigma = 0.2,n1 = 0.02,n2 = 0.01,n3 = 0.02,n4 = 0.01):
    main_data = {i_sig:n_inlier}

    if x is None:
        x = np.linspace(0,1,n_ftrs)
    X = []
    y = []
    for key,value in main_data.items():
        for _ in range(value):
            Xp = signal(key,x,sigma,n1,n2,n3,n4)
            X.append(Xp)
            y.append(key)

    for i in range(1,n_sig):
        if i!=i_sig:
            for j in range(n_outlier):
                Xp = signal(i,x,sigma,n1,n2,n3,n4)
                X.append(Xp)
                y.append(i)
            
    return np.array(X),np.array(y)

def synt_unbalanced(train_data = {1:1000,2:1000,3:1000,4:1000,5:50,6:50},
										test_data = {1:1000,2:1000,3:1000,4:1000,5:50,6:50,7:50,8:50,9:50,10:50},
										sigma = 0.1,n1 = 0.005,n2 = 0.005,n3 = 0.005,n4 = 0.005,n_ftrs = 100):

    x = np.linspace(0,1,n_ftrs)

    X = []
    y = []
    for key,value in train_data.items():
        for _ in range(value):
            Xp = signal(key,x,sigma,n1,n2,n3,n4)
            X.append(Xp)
            y.append(key)
    X_train = np.array(X)
    y_train = np.array(y)

    X = []
    y = []    
    for key,value in test_data.items():
        for _ in range(value):
            Xp = signal(key,x,sigma,n1,n2,n3,n4)
            X.append(Xp)
            y.append(key)
    X_test = np.array(X)
    y_test = np.array(y)
    
    return X_train,y_train,X_test,y_test 

def simulate_shapes(numbers = {1:10,2:10,3:10,4:10,5:10,6:10,7:10,8:10,9:10,10:10},
					sigma = 0.1,n1 = 0.005,n2 = 0.005,n3 = 0.005,n4 = 0.005,n_ftrs = 100):
    x = np.linspace(0,1,n_ftrs)
    X = []
    y = []
    for key,value in numbers.items():
        for _ in range(value):
            Xp = signal(key,x,sigma,n1,n2,n3,n4)
            X.append(Xp)
            y.append(key)
    X = np.array(X)
    y = np.array(y)
    return X,y 
#def job(X_train,X,name,n_t):
#    out = {}
#    n_ftrs = X.shape[1]

#    dim_rs ={'none':'none','AE':'AE','VAE':'VAE', 
#               'PCA':PCA(n_components=2),
#               'NMF':NMF(n_components=2), 
#               'FastICA':FastICA(n_components=2, max_iter=1000)}

#    for dim_r, value in dim_rs.items():
#        print '------------------- '+dim_r+' --------------------'

#        splitter = mce.Splitter(X_train, value, clustering)

#        outliers = [None for i in range(7)]

#        for i in range(2):
#            splitter.split(1,verbose=0,training_epochs=20)

#            outliers[i] = mce.outliers(X,splitter,metrics)

#        for i,nn in enumerate([5,10,35]):
#            lof = neighbors.LocalOutlierFactor(n_neighbors=nn)
#            lof.fit(X)
#            outliers[3+i] = -lof.negative_outlier_factor_

#        isof = IsolationForest(max_samples='auto')
#        isof.fit(X_train)
#        scores_pred = isof.decision_function(X)
#        outliers[6] = scores_pred.max()-scores_pred

#        out[dim_r] = outliers

#    with open('./res/'+name+str(n_t)+'.pkl', 'wb') as f:
#          pickle.dump(out, f)

