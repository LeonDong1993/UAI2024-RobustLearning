import numpy as np
from utmLib import utils
from pdb import set_trace

# 11-03-21 15:55

def load_data(options):
    print('Loading bike dataset ...')
    data_file = '{}/{}'.format(options.root_dir, options.data_path)
    raw = utils.read_text(data_file, header=True, splitter = ',')
    data = np.array(raw, dtype='O')

    useless_attr = [0, 11, 12, 13]
    selector = utils.notin(range(data.shape[1]), useless_attr)
    data = data[:,selector].astype(np.float32)
    return data
