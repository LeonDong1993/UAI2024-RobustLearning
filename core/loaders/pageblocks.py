import numpy as np
from pdb import set_trace

# 11-03-21 20:23

def load_data(options):
    print('Loading pageblocks dataset ...')
    data_file = '{}/{}'.format(options.root_dir, options.data_path)
    data = np.loadtxt(data_file).astype(np.float32)
    data = data[:, 0:10]

    return data


