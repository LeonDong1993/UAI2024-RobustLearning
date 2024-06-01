import numpy as np
from pdb import set_trace


def onehot_encoding(data):
    N, D = data.shape
    ret = None
    for i in range(D):
        values = np.unique(data[:,i])
        X = np.zeros(shape = (N, values.size))
        mapping = dict([(v,k) for k,v in enumerate(values)])
        idxes = list(map(mapping.__getitem__, data[:,i]))
        X[range(N),idxes] = 1

        if ret is None:
            ret = X
        else:
            ret = np.concatenate([ret, X], axis=1)
        print(ret.shape)
    return ret



