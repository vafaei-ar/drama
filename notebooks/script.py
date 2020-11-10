import matplotlib as mpl
mpl.use('agg')

import sys
import numpy as np
from drama.v1.outlier_finder import D_Drama

#example:
#python script.py data.npy AE L2 

fname = sys.argv[1]
drt = sys.argv[2]
metric = sys.argv[3]

x = np.load(fname)
d_drama = D_Drama(X_seen = x, drt_name = drt)
res = d_drama(metrics=metric,n_split=2)

scores = res['real'][metric]
out = x[np.argsort(scores)[::-1]][:10]

print(out)
