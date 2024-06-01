import glob, pdb
import numpy as np
from minilib import transform_to_mnist
import matplotlib.pyplot as plt

def load_data(options):
    print('Loading dida data.....')
    data_file = '{}/{}'.format(options.root_dir, options.data_path)
    dida_imgs = read_dida(data_file, options.num_per_class, options.with_label)
    # transfrom to mnist style
    data = options.to_mnist(dida_imgs, options.down_sample, options.normalize)
    return data


def read_dida(root_dir, num_per_class, with_label = False):
    dida_images = []
    for label in range(10):
        img_files = glob.glob(f'{root_dir}/{label}/*.jpg')
        n_img = len(img_files)
        if n_img < num_per_class:
            warnings.warn(f'In dida dataset, not enough images for class {label}, will load {n_img} images.')
        
        for i in range(min(n_img, num_per_class)):
            item = plt.imread(img_files[i])
            if with_label:
                item = (item, label)
            dida_images.append( item  )
    return dida_images
