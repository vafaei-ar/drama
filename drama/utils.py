import os
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gauss_f
from scipy.signal import argrelextrema
from scipy.spatial.distance import braycurtis,canberra,chebyshev,cityblock,correlation,minkowski,wminkowski
from sklearn.metrics import matthews_corrcoef

COLORS = ['r','b','g','y','plum','sienna','darkolivegreen',
'orange','cyan','violet','lime','darkblue','maroon','greenyellow','lightpink','thistle']

def ch_mkdir(directory):
	if not os.path.exists(directory):
		  os.makedirs(directory)

def rws_score(outliers,v,n_o=None):
    if n_o is None:
        n_o = int(outliers.sum())
    b_s = np.arange(n_o)+1
    o_ind = np.argsort(v)[::-1]
    o_ind = o_ind[:n_o]
    return 1.*np.sum(b_s*outliers[o_ind].reshape(-1))/b_s.sum()

def MCC(outliers,v,n_o=None):
    if n_o is None:
        n_o = int(outliers.sum())
    o_ind = np.argsort(v)[::-1]
    o_ind = o_ind[:n_o]
    pred = np.zeros(outliers.shape)
    pred[o_ind] = 1
    return matthews_corrcoef(outliers.astype(int),pred)

def random_choice(data,frac,axis=0):
    n_tot = data.shape[axis]
    idx = np.arange(n_tot)
    np.random.shuffle(idx)
    n1 = int(frac*n_tot)
    return np.take(data, idx[:n1], axis=axis),np.take(data, idx[n1:], axis=axis)

def standard(X):
	xmin = X.min()
	X = X-xmin
	xmax = X.max()
	X = X/xmax
	return X,xmin,xmax

def vec_fun(X,nd,n_gf_p,n_pf_p,l_f,deg):
    n_0 = X.shape[0]
    n_data = [None for i in range(n_0)]
    for i in range(n_0):
        n_data[i] = feature(X[i],nd,n_gf_p,n_pf_p,l_f,deg)

    return np.array(n_data)
    
def feature(y,nd,n_gf_p,n_pf_p,l_f,deg):    
#     y = y-y[0]
	y = (y-y.min())/(y.max()-y.min())

	n_ftrs = y.shape[0]
	xx = np.arange(n_ftrs)

	# filtered
	f_v1 = np.zeros(nd)
	#number of peaks
	f_v2 = np.zeros(2)
	# maxima G
	f_v3 = np.zeros(2*n_gf_p)
	# minima G
	f_v4 = np.zeros(2*n_gf_p)
	# maxima PF
	f_v5 = np.zeros(2*n_pf_p)
	# minima PF
	f_v6 = np.zeros(2*n_pf_p)

	##### GAUSSIAN #####
	gf = gauss_f(y, l_f)
	f_v1 = gf[np.linspace(0,99,nd).astype(int)]

	# MAXIMA
	peaks = argrelextrema(gf, np.greater)[0]
	n_p = peaks.shape[0]
	f_v2[0] = n_p

	l_min = min(n_p,n_gf_p)
	f_v3[:l_min] = xx[peaks][:l_min]
	f_v3[n_gf_p:n_gf_p+l_min] = gf[peaks][:l_min]

	# MINIMA
	peaks = argrelextrema(gf, np.less)[0]
	n_p = peaks.shape[0]
	f_v2[1] = n_p

	l_min = min(n_p,n_gf_p)
	f_v4[:l_min] = xx[peaks][:l_min]
	f_v4[n_gf_p:n_gf_p+l_min] = gf[peaks][:l_min]

	##### POLY-FIT #####
	z = np.polyfit(xx, y, deg)
	p = np.poly1d(z)
	pf = p(xx)
	# MAXIMA
	peaks = argrelextrema(pf, np.greater)[0]
	n_p = peaks.shape[0]
#	f_v2[2] = n_p

	l_min = min(n_p,n_pf_p)
	f_v5[:l_min] = xx[peaks][:l_min]
	f_v5[n_pf_p:n_pf_p+l_min] = pf[peaks][:l_min]
	# MINIMA
	peaks = argrelextrema(pf, np.less)[0]
	n_p = peaks.shape[0]
#	f_v2[3] = n_p

	l_min = min(n_p,n_pf_p)
	f_v6[:l_min] = xx[peaks][:l_min]
	f_v6[n_pf_p:n_pf_p+l_min] = gf[peaks][:l_min]

	return reduce(np.append,(f_v1,f_v2,f_v3,f_v4,f_v5,f_v6))

def feature2(X,l_f): 
    X = g_fil(X,l_f)
    n_0 = X.shape[0]
    y = np.zeros((n_0,2))
    for i in range(n_0):      
        xx = X[i,:]
        y[i,0] = np.sum(np.absolute(xx)<0.1)
        xx=(xx-xx.min())/(xx.max()-xx.min())
        xx = np.diff(xx)
        y[i,1] = np.sum(xx<0)
        
    return y
    
def shuffler(x_all,y_all,perc=0.9):
    n_sample, n_feature = x_all.shape
    n_train = int(perc*n_sample)

    tr_lst = np.arange(n_sample)
    np.random.shuffle(tr_lst)
    tr_lst =  tr_lst[:n_train]
    ts_lst = np.setdiff1d(np.arange(n_sample),tr_lst)

    x_train = x_all[tr_lst]
    y_train = y_all[tr_lst]

    x_test = x_all[ts_lst]
    y_test = y_all[ts_lst]

    return [[x_train,y_train],[x_test,y_test]]

def v2t(y):
    y_int = y.astype(int)
    n_v = y.shape[0]
    n_c = np.max(y_int)
    y_o = np.zeros((n_v,n_c+1))
    for i in range(n_v):
        y_o[i,y_int[i]]=1
    return y_o

def g_fil(Y, lf):
    n_0 = Y.shape[0]
    n_data = np.zeros(Y.shape)
    for i in range(n_0):
        n_data[i,:] = gauss_f(Y[i,:], lf)

    return n_data

def g_red(Y, lf, nd):
	n_0 = Y.shape[0]
	n_data = np.zeros((n_0,nd))
	for i in range(n_0):
		yy = gauss_f(Y[i,:], lf)
		n_data[i,:] = yy[np.linspace(0,99,nd).astype(int)]
	return n_data

def trans2(xx,Xp,nd,n_gf_p,n_pf_p,deg):
	l_f = 5
	X1 = vec_fun(xx,Xp,nd,n_gf_p,n_pf_p,l_f,deg)
	l_f = 10
	X2 = feature2(Xp,l_f)
	return np.concatenate((X1,X2),axis=1)

def smoother(X,l_f):   
    n_0 = X.shape[0]
    n_data = [None for i in range(n_0)]
    for i in range(n_0):
        y = (X[i,:]-X[i,:].min())/(X[i,:].max()-X[i,:].min())
        n_data[i] = gauss_f(y, l_f)
#        n_data[i] = y[np.linspace(0,99,nd).astype(int)]
    return np.array(n_data)

def normal(X):
    return (X-X.min())/(X.max()-X.min())

def get_batch(self, n_train,X,Y):		
		num, n_features = X.shape
		num = X.shape[0]
		indx = np.arange(num)
		np.random.shuffle(indx)
		indx = indx[:n_train]
		
		x_batch = X[indx]
		y_batch = Y[indx]
		return x_batch,y_batch

def model_analyzer(X_train,y_train,model):
    n_train = y_train.shape[0]
    n_class = y_train.shape[1]
    
    pred_train = np.zeros(n_train)
    prob_train = np.zeros((n_train,n_class))
    n_split = n_train//4999+1
    
    for inds in np.array_split(np.arange(n_train), n_split):
        prob_train[inds,:] = np.array(model.predict(X_train[inds])) 
        pred_train[inds] = np.argmax(prob_train[inds,:],axis=1)
        
    y_ture = np.argmax(y_train,axis=1)
    tpn = np.argwhere(pred_train==y_ture).reshape(-1)
    fpn = np.argwhere(pred_train!=y_ture).reshape(-1)
        
    d_max = np.array([1]+[0]*(n_class-1))
    p_max = np.sum((d_max-1./n_class)**2)
    p_outlier = (p_max-np.sum((prob_train-1./n_class)**2,axis=1))/p_max
    
    return tpn,fpn,p_outlier

def outliers_ndprob(X_train,model):
    n_train = X_train.shape[0]
    n_class = model.predict(X_train[0:1]).shape[1]

    pred_train = np.zeros(n_train)
    prob_train = np.zeros((n_train,n_class))
    n_split = n_train//5000
    
    for inds in np.array_split(np.arange(n_train), n_split):
        prob_train[inds,:] = np.array(model.predict(X_train[inds])) 
        pred_train[inds] = np.argmax(prob_train[inds,:],axis=1)
        
    d_max = np.array([1]+[0]*(n_class-1))
    p_max = np.sum((d_max-1./n_class)**2)
    p_outlier = (p_max-np.sum((prob_train-1./n_class)**2,axis=1))/p_max
    
    return p_outlier

def filter_isolated_cells(array):
	import scipy.ndimage as ndimage
	""" Return array with completely isolated single cells removed
	:param array: Array with completely isolated single cells
	:param struct: Structure array for generating unique regions
	:return: Array with minimum region size > 1
	"""

	filtered_array = np.copy(array)
	id_regions, num_ids = ndimage.label(filtered_array, structure=np.ones((3,3)))
	id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
	area_mask = (id_sizes == 1)
	filtered_array[area_mask[id_regions]] = 0
	return filtered_array

def dens_centroid(z_mu,labels,n_bins=50):
    
    centroids = np.zeros((2,2))
    for i in range(2):
        filt = labels==i
        xfs = z_mu[filt][:, 0]
        yfs = z_mu[filt][:, 1]
        xedges = np.linspace(xfs.min(),xfs.max(),n_bins)
        yedges = np.linspace(yfs.min(),yfs.max(),n_bins)
        H, xedges, yedges = np.histogram2d(xfs, yfs, bins=(xedges, yedges))

        x_cent, y_cent = np.where(H==H.max())
        x_cent, y_cent = x_cent[0], y_cent[0]
        centroids[i] = np.array([xedges[x_cent],yedges[y_cent]])

    return centroids

def dense_point(z_mu,n_bins=50):

	if z_mu.shape[0]==1:
		return z_mu
	xfs = z_mu[:, 0]
	yfs = z_mu[:, 1]

	xmin = xfs.min()
	xmax = xfs.max()
	ymin = yfs.min()
	ymax = yfs.max()

	if xmax-xmin<1e-4:
		xmax += 1e-4
	if ymax-ymin<1e-4:
		ymax += 1e-4

	xedges = np.linspace(xmin,xmax,n_bins)
	yedges = np.linspace(ymin,ymax,n_bins)
	H, xedges, yedges = np.histogram2d(xfs, yfs, bins=(xedges, yedges))

	x_cent, y_cent = np.where(H==H.max())
	x_cent, y_cent = x_cent[0], y_cent[0]
	dp = np.array([xedges[x_cent],yedges[y_cent]])

	return dp

def op_cluster(fit_predict, z_mu):
    num = z_mu.shape[0]
    n_divide = int(num/4999)+1
    y = np.zeros(num)
    for inds in np.array_split(np.arange(num), n_divide):
        y[inds] = fit_predict(z_mu[inds,:])
    return y

def flushout(style,values):
    sys.stdout.write('\r'+style % values)
    sys.stdout.flush()

class sk_convert(object):
    def __init__(self, DR):
        self.model = DR    
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

def score(olind,w_t):
    n_max = olind.shape[0]
    scr = np.zeros(n_max)
    for i in range(1,n_max):
    #     scr[i] = 1.*np.sum((np.arange(i)[::-1]+1)*w_t[olind[:i]])/(1.*i*(i+1)/2) 
        olind = olind.astype(int)
        scr[i] = 1.*np.sum(w_t[olind[:i]])/(1.*i)
    return scr

######################## METRICS
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
	x = clip(minkowski(u, v, 4),a_max=700)
	return np.exp(x)

def wL2(u, v, w):
	return wminkowski(u, v, 2, w)
def wL4(u, v, w):
	return wminkowski(u, v, 4, w)
def wexpL4(u, v, w):
	x = clip(wminkowski(u, v, 4, w),a_max=700)
	return np.exp(x)

dist_funcs = {'braycurtis':braycurtis,'canberra':canberra,'chebyshev':chebyshev,
							'cityblock':cityblock,'correlation':correlation,'L2':L2
							,'L4':L4,'expL4':expL4,'wL2':wL2,'wL4':wL4,'wexpL4':wexpL4}

def dist(metr,comp,X,w=None,w_norm=False):
	func = dist_funcs[metr]
	n_test = X.shape[0]
	n_c = comp.shape[0]
	all_dist = np.zeros(n_c)
	dist = np.zeros(n_test)
	for i in range(n_test):
		for j in range(n_c):
			if w is None:
				all_dist[j] = func(X[i],comp[j])
			else:
				if w_norm:
					w = w-w.min()+0.01
					w = w/w.max()
				all_dist[j] = func(X[i],comp[j],w[j])
		dist[i] = np.nan_to_num(np.min(all_dist))
	return dist

def diag_correction(cov):
	np.fill_diagonal(cov, clip(np.diag(cov) ,a_min=1e-4))

def Cov_mat(data):
	if data.shape[0]<2 or data.max()==data.min():
		return np.eye(data.shape[1])
	c_m = np.mean(data,axis=0)
	n_obs = data.shape[0]
	n_data = data.shape[1]
	Cov = np.zeros((n_data,n_data))
	for i in range(n_obs):
		  vec = (data[i,:]-c_m).reshape(n_data,1)
		  Cov += np.matmul(vec,vec.T)
	return Cov/n_obs

def ind2score(oi):
    num = oi.shape[0]
    score = np.zeros(num)
    score[oi[::-1]] = np.linspace(0,1,num)
    return score

def corrector(outliers):

    assert isinstance(outliers, dict),'Input should be a dictionary contains outliers using a several metrics.'	
    n = 0
    sigma = 0
    for metr,dist in outliers.iteritems():
        nd = dist.shape[0]
        dist -= dist.min()
        d_max = dist.max()
        if d_max!=0:
            dist /= dist.max()
        if (dist<0.5).sum()<nd/2:
            outliers[metr] = 1-dist
    return outliers

def dic2array(x):
    vals = []
    flds = []
    for k,v in x.iteritems():
        vals.append(v)
        flds.append((k,float))
    vals = np.array(vals).T
    vals = [tuple(i) for i in vals]
    return np.array(vals,dtype=flds)

