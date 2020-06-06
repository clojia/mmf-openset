import sys
import os.path
sys.path.insert(0, os.path.abspath("./simple-dnn"))

#Import the libraries we will need.
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy
import scipy.io
from sklearn import metrics, preprocessing, model_selection
import time
import random

from simple_dnn.util.format import UnitPosNegScale, reshape_pad, flatten_unpad
from simple_dnn.generative.discriminator import DiscriminatorDC, DiscriminatorFlat
from simple_dnn.generative.gan import MultiClassGAN, FlatGAN
from simple_dnn.generative.generator import GeneratorDC, GeneratorFlat
from simple_dnn.util.sample_writer import ImageGridWriter

from open_net import OpenNetFlat, OpenNetCNN
from openmax import OpenMaxFlat, OpenMaxCNN    

class GOpenmax:
    def __init__(self, gan_factory, openmax_factory, classifier_factory, y_dim, batch_size,
                 unpad_flatten=None):
        self.gan = gan_factory(y_dim)
        self.openmax_factory = openmax_factory
        self.y_dim = y_dim
        self.batch_size = batch_size
        if unpad_flatten is None:
            self.unpad_flatten = lambda x:x
        else:
            self.unpad_flatten = unpad_flatten
        
        assert self.y_dim == self.gan.y_dim
        assert self.y_dim == self.gan.discriminator.y_dim
        self.classifier_factory = classifier_factory
    
    def generate_filter_samples(self, y, classifier):
        # - generate samples using the GAN-Generator (maintain original class distribution)
        _, count = np.unique(np.argmax(y, axis=1), return_counts=True)
        avg_class_size = count.mean()
        class_dist = count / float(count.sum())
        gen_count = 0
        gen_x_list = []
        for _ in range(10):
            gen_y = np.random.multinomial(1, class_dist, self.batch_size)
            gen_x = self.gan.generate(ys=gen_y)
            # - use k-class classifier and select all the misclassified generated samples as unknown class.
            pred_y = classifier.predict(gen_x, reformat=False)
            incorrect_pred_mask = np.logical_not(pred_y == np.argmax(gen_y, axis=1))
            if incorrect_pred_mask.sum() > 1:
                gen_count += incorrect_pred_mask.sum()
                gen_x_list.append(self.unpad_flatten(gen_x[incorrect_pred_mask]))
                if gen_count > avg_class_size:
                    break
        
        gen_y = np.zeros((gen_count, y.shape[1]+1), dtype=int)
        gen_y[:, -1] = 1
        return np.vstack(gen_x_list), gen_y
        
    
    def fit(self, X, y, val_x=None, val_y=None):
        # - train a k-class classifier
        classifier = self.classifier_factory(self.y_dim)
        print('Fit k-class classifer')
        classifier.fit(X, y, val_x, val_y[:, :-1] if val_y is not None else None)
        # - train a Conditional GAN 
        print('Fit GAN')
        self.gan.fit(X, y, val_x, val_y)
        # - generate samples using the GAN-Generator (maintain original class distribution)
        # - use k-class classifier and select all the misclassified generated samples as unknown class.
        print('Generate fake examples.')
        fake_X, fake_y = self.generate_filter_samples(y, classifier)
        # - perform weibul distribution fitting and score calibration
        print('Fit G-openmax')
        self.g_openmax = self.openmax_factory(self.y_dim + 1)
        y_extended = np.concatenate((y, np.zeros((y.shape[0], 1), dtype=int)), axis=1)
        print('X.shape', X.shape, 'fake_X.shape', fake_X.shape)
        self.g_openmax.fit(np.vstack((X, fake_X)), np.vstack((y_extended,fake_y)))
    

    def predict_prob_open(self, X):
        """ Predicts open set class probabilities for X
        """
        pred_prob = self.g_openmax.predict_prob_open(X)
        normalized_prob = np.concatenate(
            (pred_prob[:, :-2], pred_prob[:, -2:].sum(axis=1)[:,None]), axis=1)
        normalized_prob = normalized_prob / normalized_prob.sum(axis=1)[:, None]
        return normalized_prob
        
    def predict_open(self, X):
        """ Predicts closed set class probabilities for X
        """
        open_prob = self.predict_prob_open(X)
        return np.argmax(open_prob, axis=1)

    def decision_function(self, X):
        open_prob = self.predict_prob_open(X)
        return open_prob[:, -1]
        
