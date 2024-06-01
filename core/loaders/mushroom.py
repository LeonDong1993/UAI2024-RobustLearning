import numpy as np
from pdb import set_trace
from .onehot import onehot_encoding

def load_data(options):
    print('Loading data .....')
    data_file = '{}/{}'.format(options.root_dir, options.data_path)
    data = np.loadtxt(data_file, dtype=str, delimiter=',')
    one_hot_data = onehot_encoding(data).astype(np.float32)
    return one_hot_data