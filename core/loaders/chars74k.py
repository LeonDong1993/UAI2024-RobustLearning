import glob, pdb
import numpy as np


def load_data(options):
    print('Loading chars74k data.....')
    data_file = '{}/{}'.format(options.root_dir, options.data_path)
    dida_imgs = read_dida(data_file, options.num_per_class, options.with_label)
    # transfrom to mnist style
    data = options.to_mnist(dida_imgs, options.down_sample, options.normalize)
    return data


def read_chars74k(root_dir, num_per_class, with_label = False):
    ch74_images = []
    for label in range(10):
        img_files = glob.glob('{}/Sample0{:02d}/*.png'.format(root_dir, label+1))
        n_img = len(img_files)
        if n_img < num_per_class:
            warnings.warn(f'In chars74k, not enough images for class {label}, will load {n_img} images.')
        
        for i in range(min(n_img, num_per_class)):
            item = plt.imread(img_files[i])
            if with_label:
                item = (item, label)
            ch74_images.append( item )
    return ch74_images


