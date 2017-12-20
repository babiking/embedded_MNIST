# Created by babiking on Dec.16th, 2017 @tucodec...
from utils.utils import _array_sparse_to_dense
from utils.utils import _read_MNIST_file
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from model.MNIST_classification import mnist_classification
import os
import sys


# !Check if tensorflow version == '1.4.0', API
assert tf.__version__ == '1.4.0'



# !Set system default encoding method == 'utf-8'
reload(sys)
sys.setdefaultencoding('utf-8')




# assign computation tasks excuted by GPU:'0',
# uncomment if you want to use all resources i.e. multi-GPUs and 100% memory-on-board on your server
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



# img, n_rows, n_cols = _read_MNIST_file('./data/train-images.idx3-ubyte', fmt='>IIII')
# lab, _, _ = _read_MNIST_file('./data/train-labels.idx1-ubyte', fmt='>II')
#
# pyplot.figure()
# pyplot.imshow(np.reshape(img[101,:], [n_rows, n_cols]))
# pyplot.title(['%d' % lab[101]])
# pyplot.show()

with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
        MNIST_inst = mnist_classification(sess=sess, graph=graph)

        MNIST_inst._train(img_file='./data/train-images.idx3-ubyte', lab_file='./data/train-labels.idx1-ubyte')