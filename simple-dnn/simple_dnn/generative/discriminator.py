import tensorflow as tf
import tensorflow.contrib.slim as slim


class DiscriminatorDC(object):

    def __init__(self, y_dim,
                 conv_units, 
                 hidden_units=None,
                 kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
                 d_activation_fn=tf.contrib.keras.layers.LeakyReLU,     # Conv Layers
                 f_activation_fns=tf.nn.relu,                           # Fully connected
                 dropout=False, keep_prob=0.5,
                 batch_norm=True):
        self.y_dim = y_dim
        self.batch_norm = batch_norm
        ######################## Discremenator
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
            
        self.d_activation_fn = d_activation_fn

        # Fully connected layer config
        self.hidden_units = hidden_units
        if not isinstance(f_activation_fns, list) and  self.hidden_units is not None:
            self.f_activation_fns = [f_activation_fns] * len(self.hidden_units)
        else:
            self.f_activation_fns = f_activation_fns

        ######################## Training Config
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.matching_layer = None

    def build_conv(self, x, reuse=False):
        net = x
        with slim.arg_scope([slim.conv2d], padding='SAME',
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                            activation_fn=self.d_activation_fn(alpha=0.2),
                            # weights_regularizer=slim.l2_regularizer(0.0005),
                            reuse=reuse):
            fm_layer = None
            for i, (c_unit, kernel_size, stride, padding) in enumerate(zip(
                self.conv_units, self.kernel_sizes, self.strides, self.paddings)):
                # Conv
                net = slim.conv2d(net, c_unit, kernel_size, stride=stride,
                                normalizer_fn=slim.batch_norm if self.batch_norm else None,
                                padding=padding, scope='d_conv{0}'.format(i))
                if self.matching_layer is not None and i == self.matching_layer:
                    fm_layer = net
                    
            # Dropout: Do NOT use dropout for conv layers. Experiments show it gives poor result.
        return net, fm_layer

    def build_full(self, net, reuse=False):
        with slim.arg_scope([slim.fully_connected], 
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            for i, (h_unit, activation_fn) in enumerate(zip(
                self.hidden_units, self.f_activation_fns)):
                net = slim.fully_connected(
                    net, h_unit, normalizer_fn=slim.batch_norm if self.batch_norm else None,
                    activation_fn=activation_fn,
                    reuse=reuse, scope='d_full{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
        return net
        
    def __call__(self, x, reuse=False, logits=True, matching_layer=None):
        self.matching_layer = matching_layer
        # Conv Layers
        net, fm_layer = self.build_conv(x, reuse=reuse)
        # Flatten Conv Layer Output
        net = slim.flatten(net)

        # Fully Connected Layers
        if self.hidden_units is not None:
            net = self.build_full(net, reuse=reuse)

        # Output logits
        if logits:
            d_out = slim.fully_connected(
                net, self.y_dim if self.y_dim == 1 else self.y_dim + 1, activation_fn=None, 
                reuse=reuse, scope='d_out',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        else:
            d_out = slim.fully_connected(
                net, self.y_dim if self.y_dim == 1 else self.y_dim + 1, activation_fn=tf.nn.sigmoid, 
                reuse=reuse, scope='d_out',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return d_out, fm_layer

class DiscriminatorFlat(object):
    """ Fully connected Discriminator.
    """
    def __init__(self, y_dim,
                 hidden_units,
                 f_activation_fns=tf.nn.relu,                           # Fully connected
                 dropout=False, keep_prob=0.5,
                 batch_norm=True):
        self.y_dim = y_dim
        self.batch_norm = batch_norm
        ######################## Discremenator
        # Fully connected layer config
        self.hidden_units = hidden_units
        if not isinstance(f_activation_fns, list) and  self.hidden_units is not None:
            self.f_activation_fns = [f_activation_fns] * len(self.hidden_units)
        else:
            self.f_activation_fns = f_activation_fns

        ######################## Training Config
        self.dropout = dropout
        self.keep_prob = keep_prob
        self.matching_layer = None
        self.is_training = True

    def build_full(self, net, reuse=False):
        with slim.arg_scope([slim.fully_connected], 
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            for i, (h_unit, activation_fn) in enumerate(zip(
                self.hidden_units, self.f_activation_fns)):
                net = slim.fully_connected(
                    net, h_unit, normalizer_fn=slim.batch_norm if self.batch_norm else None,
                    activation_fn=activation_fn,
                    reuse=reuse, scope='d_full{0}'.format(i))
                if self.dropout:
                    net = slim.dropout(net, keep_prob=self.keep_prob, is_training=self.is_training)
        return net
        
    def __call__(self, x, reuse=False, logits=True, matching_layer=None):
        self.matching_layer = matching_layer

        net = slim.flatten(x)

        # Fully Connected Layers
        if self.hidden_units is not None:
            net = self.build_full(net, reuse=reuse)

        # Output logits
        if logits:
            d_out = slim.fully_connected(
                net, self.y_dim if self.y_dim == 1 else self.y_dim + 1, activation_fn=None, 
                reuse=reuse, scope='d_out',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        else:
            d_out = slim.fully_connected(
                net, self.y_dim if self.y_dim == 1 else self.y_dim + 1, activation_fn=tf.nn.sigmoid, 
                reuse=reuse, scope='d_out',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return d_out, None
