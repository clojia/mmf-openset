import sys
import os.path
sys.path.insert(0, os.path.abspath("./simple-dnn"))

#Import the libraries we will need.
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
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

from open_net import  OpenNetBase

from open_net import OpenNetFlat, OpenNetCNN, OpenNetBase

class CentralOpennet(OpenNetBase):
    def __init__(self, x_dim, y_dim,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 contamination=0.02,
                 penalty=1):

        self.penalty = penalty 

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(CentralOpennet, self).__init__(
            x_dim, y_dim, z_dim=y_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale, x_reshape=x_reshape,
            opt=opt, recon_opt=None, c_opt=None, threshold_type=threshold_type,
            dist='mean_separation_spread', decision_dist_fn=decision_dist_fn, dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory,
            ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=True,
            div_loss=False, contamination=contamination)


    def loss_fn_training_op(self, x, y, z, logits, class_means):
        """ Computes the loss functions and creates the update ops.

        :param x - input X
        :param y - labels y
        :param z - z layer transform of X.
        :param logits - softmax logits if ce loss is used. Can be None if only ii-loss.
        :param recon - reconstructed X. Experimental! Can be None.
        :class_means - the class means.
        """
        # Calculate intra class and inter class distance
        self.intra_c_loss, self.inter_c_loss = self.inter_separation_intra_spred(
            z, tf.cast(y, tf.int32), class_means)
        
        intra_loss = tf.reduce_mean(self.intra_c_loss)
        
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))

        self.loss = ce_loss + (self.penalty * intra_loss)

        tvars = tf.trainable_variables()
        classifier_vars = [var for var in tvars if 'enc_' in var.name or 'classifier_' in var.name]

        # Training Ops
        self.train_op = self.opt.minimize(self.loss, var_list=classifier_vars)

    def fit(self, X, y, X_val=None, y_val=None):
        """ Fit model.
        """
        assert y.shape[1] == self.y_dim
        start = time.time()
        self.is_training = True
        count_skip = 0
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            i = 0
            while i < self.iterations:
                xs, ys = self._next_batch(X, y)
                xs = self.x_reformat(xs)

                intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss, acc, val_acc = \
                    None, None, None, None, None, None, None

                if len(np.unique(np.argmax(ys, axis=1))) != self.y_dim:
                    count_skip += 1
                    continue

                _, loss, acc = self.sess.run(
                    [self.train_op, self.loss, self.acc],
                    feed_dict={self.x:xs, self.y:ys})
                if X_val is not None and y_val is not None:
                    val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

                if i % self.display_step == 0 and self.display_step > 0:
                    self.update_class_stats(X, y)
                    acc = (self.predict(xs, reformat=False) == np.argmax(ys, axis=1)).mean()
                    if X_val is not None and y_val is not None:
                        val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

                    self._iter_stats(i, start,  intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss,
                                     acc, val_acc)
                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                    self.save_model('model-'+str(i)+'.cptk')
                    print ("Saved Model")

                i += 1
                
        
        if self.display_step > 0:
            self.update_class_stats(X, y)
            acc = (self.predict(xs, reformat=False) == np.argmax(ys, axis=1)).mean()
            if X_val is not None and y_val is not None:
                val_acc = (self.predict(X_val) == np.argmax(y_val, axis=1)).astype(np.float).mean()

            self._iter_stats(i, start,  intra_c_loss, inter_c_loss, recon_loss, loss, ce_loss,
                             acc, val_acc)
            
        if self.model_directory is not None:
            self.save_model('model-'+str(i)+'.cptk')
            print ("Saved Model")

        # Save class means and cov
        self.update_class_stats(X, y)

        # Compute and store the selected thresholds for each calls
        self.class_thresholds(X, y)
        self.is_training = False


class CentralOpennetFlat(CentralOpennet):
    def __init__(self, x_dim, y_dim, 
                 h_dims=[100],
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 contamination=0.02,
                 penalty=1):


        # Network Setting
        if isinstance(h_dims, list) or isinstance(h_dims, tuple):
            self.h_dims = h_dims
        else:
            self.h_dims = [h_dims]

        self.activation_fn = activation_fn

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(CentralOpennetFlat, self).__init__(
            x_dim, y_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale,
            x_reshape=x_reshape,
            opt=opt,
            decision_dist_fn=decision_dist_fn, threshold_type=threshold_type,
            dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory, 
            contamination=contamination,
            penalty=penalty)

    
    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            A tuple z, softmax input logits
        """
        net = x
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            for i, num_unit in enumerate(self.h_dims):
                net = slim.fully_connected(
                    net, num_unit,
                    normalizer_fn=slim.batch_norm,
                    reuse=reuse, scope='enc_{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        logits = slim.fully_connected(
            z, self.y_dim, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            scope='classifier_logits', reuse=reuse)

        return z, logits

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        self.z, logits = self.encoder(self.x)
        
        # Calculate class mean
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        self.loss_fn_training_op(self.x, self.y, self.z, logits, self.class_means)

        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))

        # For Inference, set is_training
        self.is_training = False
        self.z_test, logits_test = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=logits_test)
        self.is_training = True


class CentralOpennetCNN(CentralOpennet):
    def __init__(self, x_dim, x_ch, y_dim, 
                 conv_units=[32,64], hidden_units=[],
                 kernel_sizes=[5,5], strides=[1, 1], paddings='SAME',
                 pooling_enable=False, pooling_kernel=[2,2],
                 pooling_stride=[2,2], pooling_padding='SAME',
                 pooling_type='max', # 'avg' or 'max'
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 x_reshape=None,
                 opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                 decision_dist_fn = 'euclidean',
                 threshold_type='global',
                 dropout = True, keep_prob=0.7,
                 batch_size=128, iterations=5000,
                 display_step=500, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 contamination=0.02,
                 penalty=1):

        self.x_ch = x_ch

        # Conv layer config
        self.conv_units = conv_units
        if isinstance(kernel_sizes[0], list) or isinstance(kernel_sizes[0], tuple):
            assert len(conv_units) == len(kernel_sizes)
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_sizes] * len(conv_units)

        if isinstance(strides[0], list) or isinstance(strides[0], tuple):
            assert len(conv_units) == len(strides)
            self.strides = strides
        else:
            self.strides = [strides] * len(conv_units)

        if isinstance(paddings, list):
            assert len(conv_units) == len(paddings)
            self.paddings = paddings
        else:
            self.paddings = [paddings] * len(conv_units)

        # Conv pooling config
        self.pooling_enable = pooling_enable
        assert pooling_type in ['avg', 'max']   # supported pooling types.
        self.pooling_type = pooling_type

        if isinstance(pooling_kernel[0], list) or isinstance(pooling_kernel[0], tuple):
            assert len(conv_units) == len(pooling_kernel)
            self.pooling_kernels = pooling_kernel
        else:
            self.pooling_kernels = [pooling_kernel] * len(conv_units)

        if isinstance(pooling_stride[0], list) or isinstance(pooling_stride[0], tuple):
            assert len(conv_units) == len(pooling_stride)
            self.pooling_strides = pooling_stride
        else:
            self.pooling_strides = [pooling_stride] * len(conv_units)

        if isinstance(pooling_padding, list):
            assert len(conv_units) == len(pooling_padding)
            self.pooling_paddings = pooling_padding
        else:
            self.pooling_paddings = [pooling_padding] * len(conv_units)

        # Fully connected layer config
        self.hidden_units = hidden_units

        self.activation_fn = activation_fn

        assert decision_dist_fn in ['euclidean', 'mahalanobis']

        super(CentralOpennetCNN, self).__init__(
            x_dim, y_dim,
            x_scale=x_scale, x_inverse_scale=x_inverse_scale,
            x_reshape=x_reshape,
            opt=opt,
            decision_dist_fn=decision_dist_fn, threshold_type=threshold_type,
            dropout=dropout, keep_prob=keep_prob,
            batch_size=batch_size, iterations=iterations,
            display_step=display_step, save_step=save_step,
            model_directory=model_directory, 
            contamination=contamination,
            penalty=penalty)


        self.model_params += ['x_ch', 'conv_units', 'kernel_sizes', 'strides', 'paddings',
                              'pooling_enable', 'pooling_type', 'pooling_kernel', 'pooling_strides',
                              'pooling_padding', 'hidden_units', 'activation_fn']


    def build_conv(self, x, reuse=False):
        """ Builds the convolutional layers.
        """
        net = x
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.contrib.layers.xavier_initializer(),#tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, (c_unit, kernel_size, stride, padding, p_kernel, p_stride, p_padding) in enumerate(zip(
                    self.conv_units, self.kernel_sizes, self.strides, self.paddings,
                    self.pooling_kernels, self.pooling_strides, self.pooling_paddings)):
                # Conv
                net = slim.conv2d(net, c_unit, kernel_size, stride=stride,
                                  normalizer_fn=slim.batch_norm,
                                  reuse=reuse, padding=padding, scope='enc_conv{0}'.format(i))

                if self.display_step > 0:
                    print ('Conv_{0}.shape = {1}'.format(i, net.get_shape()))
                # Pooling
                if self.pooling_enable:
                    if self.pooling_type == 'max':
                        net = slim.max_pool2d(net, kernel_size=p_kernel, scope='enc_pool{0}'.format(i),
                                              stride=p_stride, padding=p_padding)
                    elif self.pooling_type == 'avg':
                        net = slim.avg_pool2d(net, kernel_size=p_kernel, scope='enc_pool{0}'.format(i),
                                              stride=p_stride, padding=p_padding)

                    if self.display_step > 0:
                        print ('Pooling_{0}.shape = {1}'.format(i, net.get_shape()))
                # Dropout: Do NOT use dropout for conv layers. Experiments show it gives poor result.
        return net

    def encoder(self,  x, reuse=False):
        """ Builds the network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Returns:
            Latent variables z and logits(which will be used if ce_loss is enabled.)
        """
        # Conv Layers
        net = self.build_conv(x, reuse=reuse)
        net = slim.flatten(net)

        # Fully Connected Layer
        with slim.arg_scope([slim.fully_connected], reuse=reuse,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=self.activation_fn):
            for i, h_unit in enumerate(self.hidden_units):
                net = slim.fully_connected(net, h_unit,
                                           normalizer_fn=slim.batch_norm,
                                           scope='enc_full{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training,
                                       scope='enc_full_dropout{0}'.format(i))

        # Latent Variable
        # It is very important to batch normalize the output of encoder.
        z = slim.fully_connected(
            net, self.z_dim, activation_fn=None,
            normalizer_fn=slim.batch_norm,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='enc_z')

        # It is very important to batch normalize the output of encoder.
        logits = slim.fully_connected(
            z, self.y_dim, activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='classifier_logits')

        return z, logits


    def build_model(self):
        """ Builds the network graph.
        """
        self.x = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        self.z, logits = self.encoder(self.x)

        if self.enable_recon_loss:
            self.x_recon = self.decoder(self.z)
        else:
            self.x_recon = None

        # Calculate class mean
        self.class_means = self.bucket_mean(self.z, tf.argmax(self.y, axis=1), self.y_dim)

        self.loss_fn_training_op(slim.flatten(self.x), self.y, self.z,
                                 logits, self.class_means)

        self.pred_prob = tf.nn.softmax(logits=logits)
        pred = tf.argmax(self.pred_prob, axis=1)
        actual = tf.argmax(self.y, axis=1)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, actual), tf.float32))

        # For Inference, set is_training. Can be done in a better, this should do for now.
        self.is_training = False
        self.z_test, logits_test = self.encoder(self.x, reuse=True)
        self.pred_prob_test = tf.nn.softmax(logits=logits_test)
        self.is_training = True
