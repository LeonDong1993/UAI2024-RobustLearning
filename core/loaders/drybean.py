import numpy as np
from pdb import set_trace

def load_data(options):
    print('Loading drybean data .....')
    data_file = '{}/{}'.format(options.root_dir, options.data_path)

    fh = open(data_file)
    content = fh.readlines()
    fh.close()

    st = 0
    for i,elem in enumerate(content):
        if elem.startswith('@DATA'):
            st = i + 1
            break

    content = content[st:]
    for i,elem in enumerate(content):
        content[i] = elem.strip().split(',')

    data = np.array(content, dtype = "O")
    data = data[:, 0:-1].astype(np.float32)


    return data


