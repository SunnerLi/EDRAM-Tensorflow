# https://github.com/kvn219/cluttered-mnist/blob/master/spatial-transformer-network/cluttered_mnist.py

import numpy as np
import h5py

def load(file_name='mnist_clusttered.hdf5'):
    """
        Load the MNIST Clusttered dataset and transfer into thensorflow order

        Arg:    file_name   - The name of the MNIST Clusttered dataset hdf5 file
        Ret:    The images, label and location parameters
    """
    clustter_file = h5py.File('mnist_clusttered.hdf5', 'r')
    return theanoOrder2TensorflowOrder(clustter_file['features']), \
            clustter_file['labels'], \
            clustter_file['locations']

def theanoOrder2TensorflowOrder(_tensor):
    """
        Transfer the tensor from theano order to tensorflow order

        Arg:    _tensor - The numpy object with original order
        Ret:    The numpy object with tensorflow order
    """
    return np.swapaxes(np.swapaxes(_tensor, 1, 2), 2, 3)

if __name__ == '__main__':
    load()