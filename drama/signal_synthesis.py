from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import *

def srange(x):
    x -= x.min()
    x /= x.max()
    return x

def signal(i,x,sigma,n1,n2,n3,n4):
    n_ftrs = x.shape[0]
    n = np.random.normal(0,sigma,n_ftrs)
    n1 = np.random.uniform(1-n1,1+n1)
    n2 = np.random.uniform(0-n2,0+n2)
    n3 = np.random.uniform(1-n3,1+n3)
    n4 = np.random.uniform(0-n4,0+n4)

    if i==1:
          out = x*np.sin(2*np.pi*n1*(x-n2))
          out = out+0.8
    elif i==2:
          out = sawtooth(10*np.pi*n1*(x-n2))
          out = (out+1)/2.
    elif i==3:
          out = np.heaviside(np.sin(6*np.pi*n1*(x-n2)),0.5)
    elif i==4:
          out = np.tan(n1*(x-n2))
          out = out/1.5
    elif i==5:
          out = gausspulse(0.5*n1*(x-0.5-n2), fc=5)
          out = (out+1)/2.
    elif i==6:
          out = -n1*(x-n2)+np.sin(5*np.pi*n1*(x-n2))
          out = (out+1.7)/2.7
    elif i==7:
          out = np.heaviside(np.sin(4*np.pi*n1*(x-n2)**2),0.5)
    elif i==8:
          out = np.exp(np.sin(6*np.pi*n1*(x-n2)**3))
          out = (out-0.3)/2.5
    elif i==9:
          sig = np.sin(2 * np.pi * x)
          out = square(60*np.pi*n1*(x-n2), duty=(sig + 1)/2)
          out = (out+1)/2.
    elif i==10:
          out = np.clip(np.sin(25*n1*(x-n2)),-1,0)
          out = out+1

    else:
          print ('ERROR!')
          return
    #     out = srange(out)
    out = n3*(out-n4)+n
    return out

def event_sig(signal,mu=[0,1],amp=[0.3,0.4],sig=[0.08,0.1]):
    x = np.linspace(0,1,signal.shape[0])
    mu = np.random.uniform(mu[0],mu[1])
    amp = np.random.uniform(amp[0],amp[1])
    sig = np.random.uniform(sig[0],sig[1])
    ev = amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return signal+ev 
