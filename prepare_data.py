from __future__ import print_function
import os 
from argparse import ArgumentParser
import h5py
import numpy as np


parser = ArgumentParser()
parser.add_argument('--source', type=str, help='location of .npz files', required=True)
parser.add_argument('--dataset-path', type=str, help='path of .h5 dataset', required=True)
options = parser.parse_args()
print('Preparing dataset\n{}/* --> {}'.format(options.source, options.dataset_path))

IMAGE_H, IMAGE_W = 40, 40
N_ITERS = 100

files = os.listdir(options.source)
iters_shape = (len(files), IMAGE_H, IMAGE_W, N_ITERS)
iters_chunk_shape = (1, IMAGE_H, IMAGE_W, 1)
target_shape = (len(files), IMAGE_H, IMAGE_W, 1)
target_chunk_shape = (1, IMAGE_H, IMAGE_W, 1)


with h5py.File(options.dataset_path, 'w') as h5f:
    iters = h5f.create_dataset('iters', iters_shape, chunks=iters_chunk_shape)
    targets = h5f.create_dataset('targets', target_shape, chunks=target_chunk_shape)
    
    for i, file_name in enumerate(files):
        file_path = os.path.join(options.source, file_name)
        arr = np.load(file_path)['arr_0']
        arr = arr.transpose((1, 2, 0))
        iters[i] = arr
        
        th_ = arr.mean(axis=(0, 1), keepdims=True)
        targets[i] = (arr > th_).astype('float32')[:, :, [-1]]

