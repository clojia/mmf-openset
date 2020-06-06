
""" Autoencoder Implementation.
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time


class Autoencoder(object):
  """ Variational Autoencoder Implementation.
  """
  def __init__(self, x_dim, hidden_dim,
               optimizer=tf.train.AdamOptimizer(), batch_size=128, 
               training_epochs=100, display_step=-1,
               activation_fn=tf.nn.relu, 
               output_activation_fn=None):
    self.x_dim = x_dim
    self.hidden_dim = hidden_dim
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.training_epochs = training_epochs
    self.display_step = display_step
    self.activation_fn = activation_fn
    self.output_activation_fn = output_activation_fn

    self.graph = tf.Graph()
    self.build_model()
    
    with self.graph.as_default():    
      # To save and restore all the variables.
      self.saver = tf.train.Saver()

      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())

  def encoder(self, x):
    """Construct an encoder network.
    Args:
    x: A batch of input data.
    Returns:
    net: hidden layer output
    """
    net = slim.flatten(x)
    net = slim.fully_connected(
        net, self.hidden_dim,  activation_fn=self.activation_fn,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        scope='enc')
    return net

  def decoder(self, hidden):
    """ Build a decoder network.
    Args:
    hidden: Samples of hidden layer output.
    Returns:
    net: reconstructed data
    """
    net = hidden
    net = slim.fully_connected(
        net, self.x_dim, activation_fn=self.output_activation_fn,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        scope='dec')
    return net

  def build_model(self):
    """ Build the CVAE network.
    """
    with self.graph.as_default():
      # Input placeholders
      with tf.name_scope('data'):
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        self.hidden = tf.placeholder(tf.float32, [None, self.hidden_dim])

      self.hidden = self.encoder(x=self.x)

      self.reconstructed = self.decoder(self.hidden)

      l2_norm = tf.pow(tf.subtract(self.reconstructed, self.x), 2.0)
      self.recon_cost = tf.reduce_mean(l2_norm, 1)
      self.cost = 0.5 * tf.reduce_sum(l2_norm)
      self.train_op = self.optimizer.minimize(self.cost)

  def _get_random_block_from_data(self, X, batch_size):
    start_index = np.random.randint(0, X.shape[0] - batch_size)
    return X[start_index:(start_index + batch_size)]

  def _iter_stats(self, i, start_time, loss):
    if i == 0:
        print '{0:5}| {1:8}| {2:4}'.format(
            'i', 'Loss', 'TIME')

    print '{0:5}| {1:8.4}| {2:4}s'.format(
      i, loss, int(time.time() - start_time))


  def fit(self, X):
    start_time = time.time()
    with self.graph.as_default():
        n_samples = X.shape[0]
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_X = self._get_random_block_from_data(X, self.batch_size)

                # Fit training using batch data
                cost = self._partial_fit(batch_X)
                avg_cost += cost / n_samples * self.batch_size
                
            if self.display_step > 0 and epoch % self.display_step == 0:
              self._iter_stats(epoch, start_time, avg_cost)

        if self.display_step > 0:
            self._iter_stats(epoch, start_time, avg_cost)

  def _partial_fit(self, batch_x):
    _, cost = self.sess.run((self.train_op, self.cost),
                            feed_dict={self.x: batch_x})
  
    return cost

  def transform(self, X):
    with self.graph.as_default():
      return self.sess.run(self.hidden, 
                           feed_dict={self.x: X})

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)

  def reconstruct(self, X):
    with self.graph.as_default():
      return self.sess.run(self.reconstructed, 
                           feed_dict={self.x: X})

  def reconstruct_from_hidden(self, Z):
    with self.graph.as_default():
      return self.sess.run(self.reconstructed, 
                           feed_dict={self.hidden: Z})

  def reconstruct_cost(self, X):
    with self.graph.as_default():
      return self.sess.run(self.recon_cost,
                           feed_dict={self.x: X})

  def save_model(self, model_file):
    with self.graph.as_default():
      save_path = self.saver.save(self.sess, model_file)
      print "Model saved: ", save_path
      return save_path

  def restore_model(self, model_file):
    with self.graph.as_default():
      self.saver.restore(self.sess, model_file)

class StackedAutoencoder:
  def __init__(self, x_dim, hidden_dim,
               optimizer=tf.train.AdamOptimizer(), batch_size=128, 
               training_epochs=100, display_step=-1,
               activation_fn=tf.nn.relu, output_activation_fn=None):
    self.x_dim = x_dim
    if isinstance(hidden_dim, list):
      self.hidden_dim = hidden_dim
    else:
      self.hidden_dim = [hidden_dim]
    self.optimizer = optimizer
    self.batch_size = batch_size
    self.training_epochs = training_epochs
    self.display_step = display_step
    self.activation_fn = activation_fn
    self.output_activation_fn = output_activation_fn
    
    self.stack = []
  
  def fit(self, X):
    hidden = X
    for i, dim in enumerate(self.hidden_dim):
      input_dim = hidden.shape[1]
      ae = Autoencoder(
        input_dim, dim, optimizer=self.optimizer, 
        batch_size=self.batch_size,  training_epochs=self.training_epochs, 
        display_step=self.display_step, activation_fn=self.activation_fn, 
        output_activation_fn=self.output_activation_fn if i == 0 else tf.nn.relu)
      hidden = ae.fit_transform(hidden)
      self.stack.append(ae)
  
  def transform(self, X):
    hidden = X
    for i in range(len(self.hidden_dim)):
      hidden = self.stack[i].transform(hidden)
    return hidden

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)
  
  def reconstruct(self, X):
    net = self.transform(X)
    for i in range(len(self.hidden_dim)-1, -1, -1):
      net = self.stack[i].reconstruct_from_hidden(net)
    return net


# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("data/MNIST_data/", 
#                                   one_hot=True)

# ae = Autoencoder(
#     28*28, 10,
#     optimizer=tf.train.AdamOptimizer(), 
#     batch_size=128, 
#     training_epochs=4, 
#     display_step=1,
#     activation_fn=tf.nn.relu, 
#     output_activation_fn=None)

# ae.fit(mnist.train.images)
# print ae.reconstruct(mnist.train.images).shape
# print ae.reconstruct_cost(mnist.train.images).shape
# print ae.transform(mnist.train.images).shape
# print ae.reconstruct_from_hidden(ae.transform(mnist.train.images)).shape


# sae = StackedAutoencoder(
#     28*28, [10, 5],
#     optimizer=tf.train.AdamOptimizer(), 
#     batch_size=128, 
#     training_epochs=4, 
#     display_step=1,
#     activation_fn=tf.nn.relu, 
#     output_activation_fn=None)

# sae.fit(mnist.train.images)
# print sae.reconstruct(mnist.train.images).shape
# print sae.transform(mnist.train.images).shape
