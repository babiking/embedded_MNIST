# Created by babiking on Dec.16th, 2017 @tucodec...
# MNIST classification: recognize the digit from 0 to 9...
import tensorflow as tf
import numpy as np
import struct
import random


def _read_MNIST_file(filename, fmt):

    '''
        Function:
                    _read_MNIST_file...
        Input:
                [1] <string> filename: e.g. filename of MNIST image...
        Output:
                [1] <numpy> data: dim->[num_of_samples, dim_feat]
    '''
    with open(filename, 'rb') as fin:
        buf = fin.read()

        # !Read MNIST dataset header...
        offset = 0
        try:
            [magic_number, num_of_samples, num_of_rows, num_of_cols] = struct.unpack_from(fmt, buf, offset)
        except:
            [magic_number, num_of_samples] = struct.unpack_from(fmt, buf, offset)
            num_of_rows = 1
            num_of_cols = 1

        offset += struct.calcsize(fmt)

        # !Pre-allocate memory for output data...
        dim_data = np.multiply(num_of_rows, num_of_cols)
        data = np.zeros(shape=[num_of_samples, num_of_rows, num_of_cols])

        # !Change the fmt from e.g.'>IIII' to '>784B' to read binary data...
        new_fmt = '>%sB' % dim_data


        # !Read MNIST data sample by sample...
        for ii in range(num_of_samples):

            sample = struct.unpack_from(new_fmt, buffer=buf, offset=offset)
            offset += struct.calcsize(new_fmt)

            sample = np.array(np.reshape(sample, newshape=[num_of_rows, num_of_cols]))
            data[ii, :, :] = sample


    return data, num_of_rows, num_of_cols





def _array_sparse_to_dense(sparse_vec, num_of_classes):
    '''
        Function:
                    _array_sparse_to_dense
        Input:
                [1] <tensor> sparse_vec: dim->[num_of_sample]
                [2] <int32>  num_of_class
        Output:
                <tensor> dense_vec: dim->[num_of_samples, num_of_class]
    '''
    num_of_samples = sparse_vec.shape[0]

    dense_vec = np.zeros(shape=[num_of_samples, num_of_classes])

    for ii in range(num_of_samples):
        dense_vec[ii,sparse_vec[ii]] = 1

    # for ii in range(num_of_samples):
    #     dense_vec.append(tf.sparse_to_dense(sparse_vec[ii], output_shape=(num_of_classes, ), sparse_values=1, validate_indices=False))

    return dense_vec






def _randomly_sample(inp_data, num_of_selects):
    '''
        Function:
                    _randorm_sample, i.e. randomly select N=num_of_select samples from data...
        Input:
                [1] <numpy array> inp_data
                [2] <int32>       num_of_selects
        Output:
                <numpy array> data_batch
    '''

    num_of_samples = np.size(inp_data, axis=0)

    indx = random.sample(population=range(num_of_samples), k=num_of_selects)

    data_batch = inp_data[indx, :, :]

    return np.squeeze(data_batch), indx
