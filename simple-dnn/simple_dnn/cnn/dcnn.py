import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
from sklearn import metrics
import time
import pickle

class DCNN(object):
    def __init__(self, x_dim, x_ch, y_dim, conv_units, hidden_units,
                 kernel_sizes=[5,5], strides=[1, 1], paddings='SAME',
                 pooling_enable=False, pooling_kernel=[2,2], 
                 pooling_stride=[2,2], pooling_padding='SAME',
                 activation_fns=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer,
                 iterations=100, batch_size=128,
                 display_step=10, save_step=100,
                 model_directory=None,
                 x_reformat=None,
                 dropout=True,
                 keep_prob=0.5,
                 exponential_decay=False, starter_learning_rate=0.1,
                 sess=None):
        """
        Args:
        x_dim : 2d list with the x dimentions.
        x_ch : int the channels in x.
        y_dim: int the number of classes.
        conv_units: a list with the number of channels in each conv layer.
        kernel_sizes: A list of length 2: [kernel_height, kernel_width] of all the conv layer filters.
                      Or a list of list, each list of size if size of the filter per cov layer. 
        strides: a list of tuples, each tuple holds the number stride of each conv layer.
                 or 2d list in which case all the conv layers will have the same stride.
        paddings: string or list of strings.
        pooling_enable: enable average pooling.
        pool_kernel: pooling layer kernel(window) size. 
        pool_stride= pooling layer stride size.
        pool_padding= pooling layer padding.
        hidden_units: list of ints, the nymber of units in each hidden unit in the 
                      fully connected layers.
        activation_fns: an activation function, for the fully connected hidden layers
                        or a list of activation functions, one for each hidden layer.
        x_reformat: reformats 
        dropout: enable dropout regularization.
        keep_prob: the keep probability when dropout regularization is enabled.
        exponential_decay: enable exponential learning rate decay. 
        starting_learning_rate: the exponential decay starting learning rate.
        sess: tensorflow.session
        """
        # Input config
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_ch = x_ch
        self.x_reformat = x_reformat
        
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
        
        if isinstance(activation_fns, list):
            self.activation_fns = activation_fns
        else:
            self.activation_fns = [activation_fns] * len(self.hidden_units)
            
        # Training config
        self.optimizer = optimizer
        self.iterations = iterations
        self.batch_size = batch_size
        self.display_step = display_step
        self.save_step = save_step
        self.model_directory = model_directory
        self.dropout = dropout
        self.is_training = True
        self.keep_prob = keep_prob
        self.exponential_decay =exponential_decay
        self.starter_learning_rate = starter_learning_rate
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            self.build_model()
            
            if not sess:
                self.sess = tf.Session()
            else:
                self.sess = sess

            # To save and restore checkpoints.
            self.saver = tf.train.Saver()
        
    def build_conv(self, x):
        net = x
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            for i, (c_unit, kernel_size, stride, padding, p_kernel, p_stride, p_padding) in enumerate(zip(
                    self.conv_units, self.kernel_sizes, self.strides, self.paddings,
                    self.pooling_kernels, self.pooling_strides, self.pooling_paddings)):
                # Conv
                net = slim.conv2d(net, c_unit, kernel_size, stride=stride, normalizer_fn=slim.batch_norm,
                                       padding=padding, scope='conv{0}'.format(i)) 
                # Pooling
                if self.pooling_enable:
                    net = slim.avg_pool2d(net, kernel_size=p_kernel, 
                                          stride=p_stride, padding=p_padding)
                # Dropout: Do NOT use dropout for conv layers. Experiments show it gives poor result.
        return net
    
    def build_full(self, net):
        for i, (h_unit, activation_fn) in enumerate(zip(
                self.hidden_units, self.activation_fns )):
            net = slim.fully_connected(net, h_unit, normalizer_fn=slim.batch_norm,
                                       activation_fn=activation_fn,
                                       scope='full{0}'.format(i))
            if self.dropout:
                net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
        return net
    
    def build_model(self):
        with self.graph.as_default():
            # Placeholders
#             with tf.variable_scope('data'):
            self.x = tf.placeholder(tf.float32, [None, self.x_dim[0], self.x_dim[1], self.x_ch])
            self.y = tf.placeholder(tf.float32, [None, self.y_dim])

            # Conv Layers
#             with tf.variable_scope('conv_layer'):
            net = self.build_conv(self.x)
            net = slim.flatten(net)

            # Fully Connected Layers
            net = self.build_full(net)

            # Output logits
            output_logits = slim.fully_connected(net, self.y_dim, activation_fn=None)
            self.prob = tf.nn.softmax(tf.nn.sigmoid(output_logits))

            # Cost fn
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=output_logits, labels=self.y))

            # Optimization
            if self.exponential_decay:
                global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(self.starter_learning_rate, global_step,
                                                           100000, 0.96, staircase=True)
                self.train_op = self.optimizer(learning_rate).minimize(self.cost, 
                                                                       global_step=global_step)
            else:
                self.train_op = self.optimizer().minimize(self.cost)
        
    def _next_batch(self, x, y):
        start_index = np.random.randint(0, x.shape[0] - self.batch_size)
        return x[start_index:(start_index + self.batch_size)], \
               y[start_index:(start_index + self.batch_size)]

    def _accuracy(self, val_x, val_y):
        pred_y = self.predict(val_x)
        return metrics.accuracy_score(np.argmax(val_y, axis=1), pred_y)
    
    def fit(self, X, y, val_x=None, val_y=None):
        start = time.time()
        self.is_training = True
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            for i in xrange(self.iterations):
                xs, ys = self._next_batch(X, y)
                if self.x_reformat is not None:
                    xs = self.x_reformat(xs, self.batch_size)
                
                _, cost = self.sess.run((self.train_op, self.cost),
                                        feed_dict={self.x:xs, self.y:ys})
                
                # Display and save
                if (i) % self.display_step == 0 and self.display_step > 0:
                    if val_x is not None and val_y is not None:
                        v_acc = self._accuracy(val_x, val_y)
                    print('i= {0:5}  cost= {1:7.5}  time= {2}s {3}'.format(
                        i, cost, int(time.time()-start), 
                        '' if v_acc is None else 'val_acc={:.4}'.format(v_acc)))
                if (i+1) % self.save_step == 0 and self.model_directory is not None:
                    self._save_mode(i)
            
            if val_x is not None and val_y is not None:
                v_acc = self._accuracy(val_x, val_y)
            print ('i= {0:5}  cost= {1:7.5}  time= {2}s {3}'.format(
                i, cost, int(time.time()-start), 
                '' if v_acc is None else 'val_acc={:.4}'.format(v_acc)))
            if self.model_directory is not None:
                self._save_mode(i)
        
        self.is_training = False
            
    def _save_mode(self, i):
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        self.saver.save(self.sess, self.model_directory+'/model-'+str(i)+'.cptk')
        print ("Saved Model")
    
    def restore_model(self, model_file):
        with self.graph.as_default():
            self.saver.restore(self.sess, model_file)
    
    def predict_proba(self, X, reformat=True):
        self.is_training = False
        with self.graph.as_default():
            prob = self.sess.run(
                self.prob, 
                feed_dict={self.x:self.x_reformat(X) 
                            if self.x_reformat is not None and reformat else X})
        self.is_training = True
        return prob
    
    def predict(self, X, reformat=True):
        self.is_training = False
        prob = self.predict_proba(X, reformat=reformat)
        self.is_training = True
        return np.argmax(prob, axis=1)

def mnist_reformat_pad(xs, batch_size=None):
    if batch_size is None:
        batch_size = xs.shape[0]
    xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform range between -1 and 1
    xs = np.lib.pad(xs, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad resize 32x32
    return xs

def mnist_reformat_28x28(xs, batch_size=None):
    if batch_size is None:
        batch_size = xs.shape[0]
    xs = (np.reshape(xs,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform range between -1 and 1
    return xs
