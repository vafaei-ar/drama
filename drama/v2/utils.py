from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import numpy as np

COLORS = ['r','b','g','y','plum','sienna','darkolivegreen',
'orange','cyan','violet','lime','darkblue','maroon','greenyellow','lightpink','thistle']

def save(filename,data):
    with open(filename+'.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(filename):
    handle = open(filename+'.pickle', 'rb')
    try:
        return pickle.load(handle)
    except:
        return pickle.load(handle, encoding='latin1')
        
def ch_mkdir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            print('could not make the directory!')

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

def standard(X):
    xmin = X.min()
    X = X-xmin
    xmax = X.max()
    X = X/xmax
    return X,xmin,xmax

def flushout(style,values):
    sys.stdout.write('\r'+style % values)
    sys.stdout.flush()



