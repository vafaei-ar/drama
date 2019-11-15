from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import gc
from .dae1d import *
from .cae1d import *
from .cae2d import *
from .utils import *

class Splitter(object):
    def __init__(self, X_train, reducer, clustering, z_dim=2, network_architecture=None):

        self.X_train,self.xmin,self.xmax = standard(X_train)
        self.n_samples, self.n_ftrs = X_train.shape
        self.z_dim = z_dim
        if reducer=='none':
            self.z_dim = self.n_ftrs
        if network_architecture is None:
            network_architecture = [[self.n_ftrs,self.n_ftrs//2,self.z_dim],[self.z_dim,self.n_ftrs//2,self.n_ftrs]]
        self.network_architecture = network_architecture
        self.reducer = reducer
        self.t_ind = np.arange(self.n_samples)
        self.clustering = clustering
        self.step = 0
        self.X = []
        self.z_mu = []
        self.labels = []
        self.mean_points = []
        self.mean_components = []
        self.dense_points = []
        self.dense_components = []
        self.mean_vector = []
        self.covariance = []
#        self.models = []
        self.true_indices = []

    def __del__(self):
        del self.X_train
        del self.n_samples
        del self.n_ftrs
        del self.t_ind
        del self.clustering
        del self.step
        del self.X
        del self.z_mu
        del self.labels
        del self.mean_points
        del self.mean_components
        del self.dense_points
        del self.dense_components
        del self.mean_vector
        del self.covariance
        del self.true_indices

    def call_model(self,reducer,network_architecture):
# Dense Autoencoder 1D
        if type(reducer) is str:
            reducer = reducer.upper()
            
        if reducer=='DAE1D':
            model = DenseAutoEncoder(network_architecture = network_architecture)

# Dense Variational Autoencoder 1D
        elif reducer=='DVAE1D':
            model = DenseVariationalAutoEncoder(network_architecture = network_architecture)
            
# Convolutional Autoencoder 1D
        elif reducer=='CAE1D':
            model = ConvolutionalAutoEncoder1D(input_dim,
                                               latent_dim)
            
# Convolutional Variational Autoencoder 1D
        elif reducer=='CVAE1D':
            model = ConvolutionalVariationalAutoEncoder1D(input_dim,
                                                          latent_dim)

# Convolutional Autoencoder 2D
        elif reducer=='CAE2D':
            model = ConvolutionalAutoEncoder2D(input_dim_x,
                                               input_dim_y,
                                               latent_dim)
            
# Convolutional Variational Autoencoder 2D
        elif reducer=='CVAE2D':
            model = ConvolutionalVariationalAutoEncoder2D(input_dim_x,
                                                          input_dim_y,
                                                          latent_dim)

# Without dimensionality reduction
        elif reducer=='NONE':
            model = n_model()

        else:
            try:
                model = sk_convert(reducer)
            except:
                assert 0,'Unknown transformation!\n \
You can used Scikit-learn dimensionality reduction methods like PCA, ICA or NMF.\n \
Or if you want to use a custom dimensionality reduction methods, It should be a \n \
class and has fit, transform and inverse_transform methods.'

        return model

    def reset_step(self):
        self.step = 0

    def _process(self, X_in,clustering,
                            labels=None,
                            ind=None,
                            training_epochs = 20,
                            verbose=True, 
                            save=None):

        if ind is None:
            X = X_in
        else:
            X = X_in[labels==ind]

        xmin = self.xmin
        xmax = self.xmax

        mean_points = np.zeros((2,self.z_dim))
        mean_components = np.zeros((2, self.n_ftrs))
        dense_points = np.zeros((2,self.z_dim))
        dense_components = np.zeros((2, self.n_ftrs))
        mean_vector = np.zeros((2, self.n_ftrs))
        covariance = np.array([np.eye(self.n_ftrs),np.eye(self.n_ftrs)])

        if X.shape[0]==0:
            return np.array([]),np.array([]),np.array([]),mean_points,mean_components,dense_points,dense_components,mean_vector,covariance

        if X.shape[0]==1:
            z_mu = np.array([[0.5,0.5]])
            for j in [0]:
                mean_components[j,:] = X
                if self.z_dim==2:
                    dense_components[j,:] = X
                mean_vector[j,:] = X
            return X,z_mu*xmax+xmin,np.array([0]),z_mu*xmax+xmin,mean_components*xmax+xmin,\
                            z_mu*xmax+xmin,dense_components*xmax+xmin,mean_vector,covariance

#########
        with self.call_model(self.reducer,self.network_architecture) as model:

            model.train(X, training_epochs=training_epochs, verbose=verbose)
            z_mu = model.encoder(X)

    #        label_out = self.clustering(z_mu)
            label_out = op_cluster(self.clustering, z_mu)

            for j in [0,1]:
                pts = z_mu[label_out==j]
                if pts.shape[0]!=0:
                    mean_points[j,:] = np.mean(pts,axis=0)
                    mean_components[j,:] = model.decoder((mean_points[j,:]).reshape(1,self.z_dim))

                    if self.z_dim==2:
                        dense_points[j,:] = dense_point(pts)                             
                        dense_components[j,:] = model.decoder((dense_points[j,:]).reshape(1,self.z_dim))
    #############################################

            for j in [0,1]:
                X_sub = X[label_out==j]*xmax+xmin
                if X_sub.shape[0]>1:
                    mean_vector[j,:] = np.mean(X_sub,axis=0)
                    covariance[j,:] = Cov_mat(X_sub)

                elif X_sub.shape[0]==1:
                    mean_vector[j,:] = X_sub[0]

            if save is not None:
                try:
                    model.save(save)
                except:
                    print ('Dimensionality reduction method has no save attribute!')


        gc.collect()

        return X,z_mu*xmax+xmin,label_out,mean_points*xmax+xmin,mean_components*xmax+xmin,\
                        dense_points*xmax+xmin,dense_components*xmax+xmin,mean_vector,covariance #,model
        

    def split(self, n_s, 
                        clustering=None,
                        training_epochs=20, 
                        verbose=True, 
                        save=None):

        if clustering is None:
            clustering = self.clustering

        k_max = n_s
        if self.step==0:

            self.step += 1
            if verbose:
                print ('Split level:',self.step)

            if save is not None:
                name = 'level'+str(self.step)
                if save[-1]=='/':
                    save_name = save+name+'_model.ckpt'
                else:
                    save_name = save+'/'+name+'_model.ckpt'
            else:
                save_name = None

            X,z_mu,labels_out,mean_points,mean_components,dense_points,dense_components,mean_vector,covariance = \
                                self._process(self.X_train,clustering,
                                                         training_epochs=training_epochs,
                                                         verbose=verbose, save=save_name)

            self.X = [X]
            self.z_mu = [z_mu]
            self.lbls = [labels_out]
            self.labels = [label_maker(self.lbls)]
            self.mean_points = [mean_points.reshape(1,2,self.z_dim)]
            self.mean_components = [mean_components.reshape(1,2,self.n_ftrs)]
            self.dense_points = [dense_points.reshape(1,2,self.z_dim)]
            self.dense_components = [dense_components.reshape(1,2,self.n_ftrs)]
            self.mean_vector = [mean_vector.reshape(1,2,self.n_ftrs)]
            self.covariance = [covariance.reshape(1,2,self.n_ftrs,self.n_ftrs)]
#            self.models = [model]
            self.true_indices = [self.t_ind[labels_out==0],self.t_ind[labels_out==1]]


            ngc = gc.collect()
            k_max -= 1

        if self.step!=0:
            k = 0
            while True:
                k += 1
                if (k>k_max):
                    break
                self.step += 1
                if verbose:
                    print ('--------------------------')
                    print ('Split level: ',self.step)
                    
                X0 = []
                z_mu0 = []
                lbls0 = []
                mean_points0 = []
                mean_components0 = []
                dense_points0 = []
                dense_components0 = []
                mean_vector0 = []
                covariance0 = []
#                models0 = []
                t_ind0 = []
                    
                for j in range(len(self.X)):
                    for ind in range(2):

                        if save is not None:
                            name = 'level'+str(self.step)+'_c'+str(2*j+ind)
                            if save[-1]=='/':
                                save_name = save+name+'_model.ckpt'
                            else:
                                save_name = save+'/'+name+'_model.ckpt'
                        else:
                            save_name = None

                        Xp,z_mu,labels_out,mean_points,mean_components,dense_points,dense_components,mean_vector,covariance = \
                                                                    self._process(
                                                                    self.X[j],
                                                                    clustering,
                                                                    labels = self.lbls[j],
                                                                    ind = ind,
                                                                    training_epochs=training_epochs,
                                                                    verbose = verbose,
                                                                    save=save_name)
                        
                        X0.append(Xp)
                        z_mu0.append(z_mu)
                        lbls0.append(labels_out)
                        mean_points0.append(mean_points)
                        mean_components0.append(mean_components)
                        dense_points0.append(dense_points)
                        dense_components0.append(dense_components)
                        mean_vector0.append(mean_vector)
                        covariance0.append(covariance)
#                        models0.append(model)
                        t_ind0.append(self.true_indices[2*j+ind][labels_out==0])
                        t_ind0.append(self.true_indices[2*j+ind][labels_out==1])

                self.X = X0
                self.z_mu = z_mu0
                self.lbls = lbls0
                self.labels.append(label_maker(self.lbls))
                self.mean_points.append(np.array(mean_points0))
                self.mean_components.append(np.array(mean_components0))
                self.dense_points.append(np.array(dense_points0))
                self.dense_components.append(np.array(dense_components0))
                self.mean_vector.append(np.array(mean_vector0))
                self.covariance.append(np.array(covariance0))
#                self.models = models0
                self.true_indices = t_ind0
        
                ngc = gc.collect()

#    def mean_components(self):
#        points = np.zeros((2**(self.step),2))
#        components = np.zeros((2**(self.step), self.n_ftrs))
#        for i,z in enumerate(self.z_mu):
#            for j in [0,1]:
#                pts = z[self.lbls[i]==j]
#                if pts.shape[0]!=0:
#                    points[2*i+j,:] = np.mean(pts,axis=0)
#                    components[2*i+j,:] = self.models[i].decoder((points[2*i+j,:]).reshape(1,2))
#                else:
#                    points[2*i+j,:] = np.array([0,0])
#                    components[2*i+j,:] = np.zeros(self.n_ftrs)
#        return points,components


#    def dense_components(self):
#        points = np.zeros((2**(self.step),2))
#        components = np.zeros((2**(self.step), self.n_ftrs))
#        for i,z in enumerate(self.z_mu):
#            for j in [0,1]:
#                pts = z[self.lbls[i]==j]
#                if pts.shape[0]!=0:
#                    points[2*i+j,:] = np.mean(pts,axis=0)
#                    components[2*i+j,:] = self.models[i].decoder((points[2*i+j,:]).reshape(1,2))
#                else:
#                    points[2*i+j,:] = np.array([0,0])
#                    components[2*i+j,:] = np.zeros(self.n_ftrs)
#        return points,components

def label_maker(labels):
    lbls = np.array([])
    for i,lst in enumerate(labels):
        lbls = np.append(lbls,lst+2*i)
    return lbls

class n_model(object):
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

