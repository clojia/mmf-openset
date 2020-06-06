import os.path
# import sys
# sys.path.insert(0, os.path.abspath("../"))

# Import the libraries we will need.
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time

from simple_dnn.util.format import UnitPosNegScale, reshape_pad


class VariationalAutoencoder(object):
    """ Conditional Variational Autoencoder
    """

    def __init__(self, x_dim, y_dim=None,
                 z_dim=64, h_dims=[128],
                 activation_fn=tf.nn.relu,
                 x_scale=UnitPosNegScale.scale,
                 x_inverse_scale=UnitPosNegScale.inverse_scale,
                 optimizer=tf.train.AdamOptimizer(),
                 batch_size=128, iterations=2000,
                 display_step=100, save_step=500,
                 model_directory=None,  # Directory to save trained model to.
                 ):
        """
        Args:
        :param x_dim - dimension of the input 
        :param y_dim - numbe of classes and the VAE will be conditional. 
                        If None the VAE will not be conditional 
        :param z_dim - the number of latent variables.
        :param h_dims - an int of a list; number of units int he hidden layers of the 
                        encoder network. The decoder network will simpley be the reverse.
        activation_fn - activation function used for the hidden layers of the encoder/decoder.
        x_scale - an input scaling function. Scale to rangeof [-1, 1].
        x_inverse_scale - reverse scaling fn. from scale of [-1, 1] to original input scale.
        optimizer - optimizer object.
        batch_size - training barch size. 
        iterations - number of training iterations.
        display_step - training info displaying interval.  
        save_step - model saving interval.
        model_directory - derectory to save model in.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.x_scale = x_scale
        self.x_inverse_scale = x_inverse_scale

        # Network Setting
        if isinstance(h_dims, list) or isinstance(h_dims, tuple):
            self.h_dims = h_dims
        else:
            self.h_dims = [h_dims]

        self.z_dim = z_dim

        self.activation_fn = activation_fn

        # Training Config
        self.batch_size = batch_size
        self.iterations = iterations
        self.display_step = display_step
        self.save_step = save_step
        self.model_directory = model_directory

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf.Session()
            self.build_model()

            # Optimizer
            self.optimizer = optimizer.minimize(self.cost)

            # To save and restore all the variables.
            self.saver = tf.train.Saver()

    def encoder(self, x, reuse=False):
        """ Encoder network.
        Args:
            :param x - input x.
            :param reuse - whether to reuse old network on create new one.
        Retuns:
            A tuple z_mu, z_log_sig
        """
        net = x
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            for i, num_unit in enumerate(self.h_dims):
                net = slim.fully_connected(
                    net, num_unit,
                    reuse=reuse, scope='enc_{0}'.format(i))

            z_mu = slim.fully_connected(
                net, self.z_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, scope='enc_mu')
            z_log_sig = slim.fully_connected(
                net, self.z_dim, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                reuse=reuse, scope='enc_log_sig')

        return z_mu, z_log_sig

    def decoder(self, z, y=None, reuse=False):
        """ Decoder Network.
        Args:
            :param z - latent variables z.
            :param y - in conditional VAE setting the class labels.
            :param reuse - whether to reuse old network on create new one.
        Retuns:
            The reconstructed x
        """
        if y is None:
            net = z
        else:
            net = tf.concat(axis=1, values=[z, y])

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            activation_fn=self.activation_fn):
            h_dims_revese = [self.h_dims[i]
                             for i in range(len(self.h_dims) - 1, -1, -1)]
            for i, num_unit in enumerate(h_dims_revese):
                net = slim.fully_connected(
                    net, num_unit,
                    reuse=reuse, scope='dec_{0}'.format(i))

        dec_out = slim.fully_connected(
            net, self.x_dim, activation_fn=tf.nn.tanh,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            reuse=reuse, scope='dec_out')
        return dec_out

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        if self.y_dim is None:
            self.y = None
        else:
            self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # Build Network
        z_mu, z_log_sigma = self.encoder(self.x)

        eps = tf.random_normal(shape=tf.shape(z_mu))
        self.z = z_mu + tf.exp(z_log_sigma / 2) * eps

        self.x_recon = self.decoder(self.z, self.y)

        # Compute Loss
        recon_loss = 0.5 * \
            tf.reduce_sum(tf.squared_difference(self.x_recon, self.x), 1)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma
                                       - tf.square(z_mu)
                                       - tf.exp(z_log_sigma), 1)

        self.cost = tf.reduce_mean(recon_loss + kl_loss)

    def _next_batch(self, x, y=None):
        start_index = np.random.randint(0, x.shape[0] - self.batch_size)
        if y is None:
            return x[start_index:(start_index + self.batch_size)], None
        else:
            return x[start_index:(start_index + self.batch_size)], \
                y[start_index:(start_index + self.batch_size)]

    def _iter_stats(self, i, start_time, loss):
        if i == 0:
            print('{0:5}| {1:8}| {2:4}'.format(
                'i', 'Loss', 'TIME'))

        print('{0:5}| {1:8.4}| {2:4}s'.format(
            i, loss, int(time.time() - start_time)))

    def fit(self, X, y=None):
        start = time.time()
        self.is_training = True
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            # Loop over all batches
            for i in range(self.iterations):
                xs, ys = self._next_batch(X, y)
                xs = self.x_scale(xs)
                # Fit training using batch data
                if self.y_dim is None:
                    feed_dict = {self.x: xs}
                else:
                    feed_dict = {self.x: xs, self.y: ys}

                cost, _ = self.sess.run(
                    (self.cost, self.optimizer), feed_dict=feed_dict)

                if i % self.display_step == 0:
                    self._iter_stats(i, start, cost)
                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                    self.save_model('model-' + str(i) + '.cptk')
                    print("Saved Model")

            self._iter_stats(i, start, cost)
            if self.model_directory is not None:
                self.save_model('model-' + str(i) + '.cptk')
                print("Saved Model")

    def generate(self, n_samples=1, zs=None, ys=None):
        with self.graph.as_default():
            if ys is None and self.y_dim is not None:
                ys = np.zeros(shape=[n_samples, self.y_dim])
                ys[np.arange(n_samples), np.random.randint(
                    0, self.y_dim, n_samples)] = 1.
            elif ys is not None:
                n_samples = len(ys)

            if zs is None:
                zs = self.sess.run(tf.random_normal([n_samples, self.z_dim]))

            if self.y_dim is None:
                feed_dict = {self.z: zs}
            else:
                feed_dict = {self.z: zs, self.y: ys}
            recon = self.sess.run(self.x_recon, feed_dict=feed_dict)

        return self.x_inverse_scale(recon)

    def save_model(self, model_file_name):
        if self.model_directory is None:
            return 'ERROR: Model directory is None'
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        return self.saver.save(self.sess, os.path.join(self.model_directory, model_file_name))

    def restore_model(self, model_file):
        with self.graph.as_default():
            self.saver.restore(self.sess, model_file)
