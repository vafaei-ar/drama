import sys
import numpy as np
from util import *

metrics = ['cityblock','L2','L4','braycurtis',
           'canberra','chebyshev','correlation','mahalanobis',
           'wL2','wL4']

def ind2score(oi):
	num = oi.shape[0]
	score = np.zeros(num)
	score[oi[::-1]] = np.linspace(0,1,num)
	return score

def outliers(X,splitter,metrics):

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

		distance_test[i] = np.min(dst_arr)
		distance['mahalanobis'] = distance_test
		      
	for metric in metrics:
		if metric in nmetrics:
		  distance_test = dist(metric,np.array(components),X)
		  distance[metric] = distance_test
		elif metric in wmetrics:
		  distance_test = dist(metric,np.array(components),X,w)
		  distance[metric] = distance_test

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

		distance_test[i] = np.min(dst_arr)
		distance['mahalanobis'] = distance_test

	for metric in metrics:
		if metric in nmetrics:
		  distance_test = dist(metric,np.array(components),Z)
		  distance[metric] = distance_test
		elif metric in wmetrics:
		  distance_test = dist(metric,np.array(components),Z,w)
		  distance[metric] = distance_test

	return distance
