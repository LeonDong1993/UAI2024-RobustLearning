import numpy as np
from pdb import set_trace

# 11-03-21 20:23

def load_data(options):
    print('Loading facebook comment dataset ...')
    data_dir = '{}/{}'.format(options.root_dir, options.data_path)
    train = np.loadtxt('{}/Train_1.csv'.format(data_dir), delimiter=',').astype(np.float32)

    items = []
    for i in range(1,10):
        items.append(np.loadtxt('{}/Test_Case_{}.csv'.format(data_dir,i), delimiter=','))
    test = np.vstack(items).astype(np.float32)

    return train, test


