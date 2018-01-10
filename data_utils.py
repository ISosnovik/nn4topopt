from keras.utils import GeneratorEnqueuer
import numpy as np
import h5py


def random_d4_transform(x_batch, y_batch):
    '''Apply random transformation from D4 symmetry group
    # Arguments
        x_batch, y_batch: input tensors of size `(batch_size, height, width, any)`
    '''
    batch_size = x_batch.shape[0]
    
    # horizontal flip
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = x_batch[mask, :, ::-1]
    y_batch[mask] = y_batch[mask, :, ::-1]

    # vertical flip
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = x_batch[mask, ::-1]
    y_batch[mask] = y_batch[mask, ::-1]
    
    # 90* rotation
    mask = np.random.random(size=batch_size) > 0.5
    x_batch[mask] = np.swapaxes(x_batch[mask], 1, 2)
    y_batch[mask] = np.swapaxes(y_batch[mask], 1, 2)

    return x_batch, y_batch


def minibatch_generator(dataset_path, size, iter_sampler):
    '''Yield batches from the dataset
    # Arguments
        dataset_path: path to `.h5` dataset
        size: batch size
        iter_sampler: callable. Should return iteration to be sliced
    # Output
        x_batch, y_batch
        
        x_batch: input tensor of size `(size, height, width, 2)`
        y_batch: output mask of size `(size, height, width, 1)`
    '''
    
    with h5py.File(dataset_path, 'r') as h5f:
        X = h5f['iters']
        Y = h5f['targets']
        
        n_obj = len(X)
        indices = np.arange(n_obj)
        np.random.shuffle(indices)

        for idx in range(0, n_obj - size + 1, size):
            excerpt = sorted(indices[idx:idx + size])
            iter_ = iter_sampler()
            
            x1 = X[excerpt, :, :, iter_]
            x2 = X[excerpt, :, :, iter_ - 1]
            x = np.stack((x1, x1 - x2), -1)
            
            y = Y[excerpt, :, :, :]

            x, y = random_d4_transform(x, y)
            yield x, y
            
            
def DatasetIterator(dataset_path, size, iter_sampler, 
                     use_multiprocessing=False, 
                     workers=1, max_queue_size=10):
    '''Iterate over the dataset
    # Arguments
        dataset_path: path to `.h5` dataset
        size: batch size
        iter_sampler: callable. Should return iteration to be sliced
        use_multiprocessing: use multiprocessing for workers, Bool
        workers: number of workers, int
        max_queue_size: maximum queue size, int
    # Output
        x_batch, y_batch
        
        x_batch: input tensor of size `(size, height, width, 2)`
        y_batch: output mask of size `(size, height, width, 1)`
    '''
    
    core_gen = minibatch_generator(dataset_path=dataset_path, size=size, iter_sampler=iter_sampler)
    enqueuer = GeneratorEnqueuer(generator=core_gen, use_multiprocessing=use_multiprocessing)
    enqueuer.start(workers, max_queue_size=max_queue_size)
    generator = enqueuer.get()
    
    while True:
        try:
            yield next(generator)
        except StopIteration:
            return    

        
