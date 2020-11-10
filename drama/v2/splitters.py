from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import sys
import numpy as np
from .utils import standard


class none_model(object):
    def __init__(self):
        pass
    def train(self,X, training_epochs=None, verbose=None):
        pass
    def encoder(self,X):
          return X
    def decoder(self,z):
          return z
    def __enter__(self):
          return self
    def __exit__(self, type, value, traceback):
          pass

class sk_convert(object):
    def __init__(self, DR):
        self.model = DR
        check = np.all(list(map(lambda x:hasattr(DR, x), ['fit','transform','inverse_transform'])))
        assert check, 'Not an appropriate sklearn model!'
    def train(self,X, training_epochs=None, verbose=None):
        self.model.fit(X)
    def encoder(self,X):
        return self.model.transform(X)
    def decoder(self,z):
        return self.model.inverse_transform(z)
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        del self.model
        
def call_model(reducer,n_ftrs,z_dim):
    network_architecture = [[n_ftrs,n_ftrs//2,z_dim],[z_dim,n_ftrs//2,n_ftrs]]
# Autoencoder
    if reducer=='AE': 
        from .AE import AutoEncoder
        model = AutoEncoder(network_architecture)
    elif reducer=='VAE':
        from .VAE import VariationalAutoEncoder
        model = VariationalAutoEncoder(network_architecture)
    elif reducer=='none': model = none_model()
    elif reducer=='PCA': 
        from sklearn.decomposition import PCA
        model = sk_convert(PCA(n_components=z_dim))
    elif reducer=='NMF':
        from sklearn.decomposition import NMF
        model = sk_convert(NMF(n_components=z_dim)) 
    elif reducer=='FastICA': 
        from sklearn.decomposition import FastICA
        model = sk_convert(FastICA(n_components=z_dim, max_iter=1000))
    elif reducer=='UMAP':
        from umap import UMAP
        model = sk_convert(UMAP(n_components=z_dim,init='random'))
    else:
        try:
            model = sk_convert(reducer)
        except:
            assert 0,'Unknown transformation!\n \
                      Scikit-learn dimensionality reduction algorithms like PCA, ICA or NMF are available.\n \
                      Or if you want to use a custom dimensionality reduction methods, It should be a \n \
                      class which has fit, transform and inverse_transform methods.'
    return model

def op_cluster(fit_predict, z_mu, chunk):
    num = z_mu.shape[0]
    n_divide = int(num/chunk)+1
    y = np.zeros(num)
    for inds in np.array_split(np.arange(num), n_divide):
        y[inds] = fit_predict(z_mu[inds,:])
    return y

def find_rprs(zmu):
    return np.mean(zmu,axis=0)

def cluster_diff(X0,X1,cl=5,ignore_perc=70):
    xm0,xd0,xu0 = np.percentile(X0,50,axis=0),np.percentile(X0,cl,axis=0),np.percentile(X0,100-cl,axis=0)
    xm1,xd1,xu1 = np.percentile(X1,50,axis=0),np.percentile(X1,cl,axis=0),np.percentile(X1,100-cl,axis=0)
    delta0,delta1 = np.sum(xu0-xd0),np.sum(xu1-xd1)
    if delta0<delta1:
        telorantce = xu0-xd0
    else:
        telorantce = xu1-xd1
    diff = np.absolute(xm0-xm1)/telorantce
    np.mean(diff)
    return np.mean(diff[diff>np.percentile(diff,ignore_perc)])

def is_cluster_done(X0,X1,diff_t=0.2,min_cluster_pop=2,cl=5,ignore_perc=70):
    if np.min([len(X0),len(X1)])<min_cluster_pop: 
        return True
    diff = cluster_diff(X0,X1,cl=cl,ignore_perc=ignore_perc)
    if diff<diff_t:
        return True
    return False

class Splitter(object):
    def __init__(self, X, reducer,
                 clustering=None,
                 z_dim=2,
                 chunk = 4999):
        
        self.X,self.xmin,self.xmax = standard(X)
        self.n_samples, self.n_ftrs = X.shape
        self.labels = np.zeros(self.n_samples)
        self.z_dim = z_dim
        self.z_mu = np.zeros((self.n_samples,z_dim))
        self.chunk = chunk
        self.done_cluster = {0:False}
        self.rprs = {}
        
        if reducer=='none':
            if self.z_dim == n_ftrs:
                pass
            else:
                self.z_dim = n_ftrs
                print('Latent dimension should be equal to number of features!')
                print('setting z_dim to {}'.format(n_ftrs))
        
        self.reducer = reducer
        
        if clustering is None:
            from sklearn.cluster import AgglomerativeClustering
            agg = AgglomerativeClustering()
            clustering = agg.fit_predict
        self.clustering = clustering
        self.n_clusters = 1
        

    def _split(self,X):
        with call_model(self.reducer,self.n_ftrs,self.z_dim) as model:
            model.train(X)
            z_mu = model.encoder(X)
            label = op_cluster(self.clustering, z_mu, self.chunk)
            
#            filt = label==0
            # TODO : need scale correction
#            rprst = model.decoder(find_rprs(z_mu).reshape(1,self.z_dim))
#            rprs1 = model.decoder(find_rprs(z_mu[filt]).reshape(1,self.z_dim))
#            rprs2 = model.decoder(find_rprs(z_mu[np.logical_not(filt)]).reshape(1,self.z_dim))

        return z_mu,label#,rprst,rprs1,rprs2

    def split(self,ns=5,dcc=None):
        
        if np.all(list(self.done_cluster.values())):
            print('There is no undone clusters.')
            return self.labels,self.rprs
            
        if dcc is None:
            dcc = is_cluster_done
            
        for _ in range(ns):
        
            if np.all(list(self.done_cluster.values())):
                print('There is no undone clusters.')
                return self.labels,self.rprs
            
            labelsp = self.labels+0
            print('==========================')
            print(np.unique(labelsp))
            
            for i in range(self.n_clusters):
                if self.done_cluster[i]: continue
                inds = np.argwhere(self.labels==i).reshape(-1)
                Xp = self.X[inds]
#                 z_mu,label,rprst,rprs1,rprs2 = self._split(Xp)
                z_mu,label = self._split(Xp)
                
                filt = label==0
                X0 = Xp[filt]
                X1 = Xp[np.logical_not(filt)]
                add_new_cluster = 0
                
                if dcc(X0,X1):
                    self.done_cluster[i] = True
#                     self.rprs[i] = rprst
                    print('{} is done!'.format(i))
                    continue
                
                self.z_mu[inds] = z_mu
                new_label = int(np.max(labelsp)+1)
                labelsp[inds[np.logical_not(filt)]] = new_label
                print('New cluster, {}, is added!'.format(new_label))
                self.done_cluster[new_label] = False
                add_new_cluster += 1
                
#                 self.rprs[i] = rprs1
#                 self.rprs[new_label] = rprs2
                
            self.labels = labelsp+0     
            self.n_clusters = len(np.unique(self.labels))
            
        undone = np.logical_not(list(self.done_cluster.values()))
        if np.any(undone):
            print('There is still {} undone clusters.'.format(np.sum(undone)))
        return self.labels,self.rprs















