import datetime
import gzip
import os.path
#import cPickle
import pickle
import sys
import time

import numpy as np
import scipy.io
from sklearn import preprocessing
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

sys.path.insert(0, os.path.abspath("./simple-dnn"))

from simple_dnn.util.format import UnitPosNegScale, reshape_pad
from util.openworld_sim import OpenWorldSim, OpenWorldMsData

def save_pickle_gz(obj, filename, protocol=1):
    """Saves a compressed object to disk
    """
    with gzip.GzipFile(filename, 'wb') as fout:
        pickle.dump(obj, fout, protocol)

def load_pickle_gz(filename):
    """Loads a compressed object from disk
    """
    with gzip.GzipFile(filename, 'rb') as fin:
        obj = pickle.load(fin)

    return obj

def load_open_dataset(dataset_name, tr_classes, seed, normalize=False):
    if dataset_name == 'ms':
        # Load Data
        with open('./data/pickle/5-bit-cluster_32-bit_minhash_1_1-gram_all-files_run1.pkl') as fin:
            data_fcg_xs, data_fcg_ys = pickle.load(fin)

        return OpenWorldMsData(data_fcg_xs, data_fcg_ys, tr_classes=tr_classes,
                               comb_val_test=False, seed=seed, normalize=normalize)

    elif dataset_name == 'msadjmat':
        with open('./data/pickle/5-bit-cluster_32-bit_minhash_1_1-gram_all-files_run1_adj_matrix.pkl') as fin:
            fcg_adj_xs, fcg_adj_ys = pickle.load(fin)

        # Flatten
        fcg_adj_xs = np.reshape(fcg_adj_xs, [fcg_adj_xs.shape[0], reduce(lambda x, y: x*y, fcg_adj_xs.shape[1:])])


        # One hot encode
        enc = preprocessing.OneHotEncoder(n_values=9, sparse=False)
        enc.fit(fcg_adj_ys.reshape(-1, 1))
        fcg_adj_ys = enc.transform(fcg_adj_ys.reshape(-1, 1))

        return OpenWorldMsData(fcg_adj_xs, np.argmax(fcg_adj_ys, axis=1),
                               tr_classes=6, comb_val_test=False, seed=1, normalize=normalize)

    elif dataset_name == 'android':
        with open('./data/pickle/android_malware_genome_project_5-bit-cluster_32-bit_minhash-1_1-gram.pkl') as fin:
            android_fcg_xs, android_fcg_ys = pickle.load(fin)

        uvalues, ucounts =  np.unique(android_fcg_ys, return_counts=True)
        min_class_size = 40  ## Smallest class size allowed

        true_classes = set(uvalues[ucounts >= min_class_size])
        mask = [False] * len(android_fcg_ys)
        for i in xrange(len(android_fcg_ys)):
            if android_fcg_ys[i] in true_classes:
                mask[i] = True

        mask = np.array(mask)

        # Only consider classes which have > min_class_size of number instance per class
        android_fcg_xs = android_fcg_xs[mask]
        android_fcg_ys = android_fcg_ys[mask]

        # Remap class labels to range between 0-8
        android_label_lookup = {c:i for i, c in enumerate(np.unique(android_fcg_ys))}
        for i in xrange(len(android_fcg_ys)):
            android_fcg_ys[i] = android_label_lookup[android_fcg_ys[i]]

        return OpenWorldMsData(android_fcg_xs, android_fcg_ys,
                               tr_classes=tr_classes,
                               comb_val_test=False, seed=seed, normalize=normalize)
    elif dataset_name == 'mnist':
        mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
        return OpenWorldSim(mnist.train.images, mnist.train.labels,
                            val_data=mnist.validation.images, val_label=mnist.validation.labels,
                            test_data=mnist.test.images, test_label=mnist.test.labels,
                            tr_classes=tr_classes, seed=seed)
    elif dataset_name == 'svhn':
        svhn_train = scipy.io.loadmat('data/SVHN_data/train_32x32.mat')
        svhn_test = scipy.io.loadmat('data/SVHN_data/test_32x32.mat')
        svhn_extra = scipy.io.loadmat('data/SVHN_data/extra_32x32.mat')

        def rotate_axis(X):
            X = np.swapaxes(X,2,3)
            X = np.swapaxes(X,1,2)
            X = np.swapaxes(X,0,1)
            return X

        def flatten(X):
            return X.reshape((-1, reduce((lambda x, y: x*y), X.shape[1:])))

        svhn_train['X'] = flatten(rotate_axis(svhn_train['X']))
        svhn_extra['X'] = flatten(rotate_axis(svhn_extra['X']))
        svhn_test['X'] = flatten(rotate_axis(svhn_test['X']))

        enc = preprocessing.OneHotEncoder( sparse=False) # n_values=10,
        enc.fit(svhn_train['y'].reshape(-1, 1))

        svhn_train['y'] = enc.transform(svhn_train['y'].reshape(-1, 1))
        svhn_extra['y'] = enc.transform(svhn_extra['y'].reshape(-1, 1))
        svhn_test['y'] = enc.transform(svhn_test['y'].reshape(-1, 1))

        return OpenWorldSim(svhn_train['X'], svhn_train['y'],
                            val_data=svhn_extra['X'][-10000:],
                            val_label=svhn_extra['y'][-10000:],
                            test_data=svhn_test['X'], test_label=svhn_test['y'],
                            tr_classes=tr_classes, seed=seed)
