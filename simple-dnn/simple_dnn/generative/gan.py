import sys
import os.path
# sys.path.insert(0, os.path.abspath("./simple-dnn"))

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import scipy.misc
import time

class BaseGAN(object):
    """ Base class for Generative Adversarial Network implementation.
    """
    def __init__(self, 
                 x_dims, x_ch, y_dim,
                 generator=None,     # Generator Net
                 discriminator=None, # Discriminator Net
                 x_reshape=None,
                 x_scale=None,
                 x_inverse_scale=None,
                 z_dim=100,
                 d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 d_label_smooth=0.75,
                 batch_size=128, iterations=2000,
                 display_step=100, save_step=1000,
                 oracle=None,
                 graph=None, sess=None,
                 sample_writer=None, # Object of SampleWriter class.  
                 model_directory=None, #Directory to save trained model to.
                ):
        """
        Args:
        x_dims - list; the width of and hight of image x.
        x_ch - int; the number of channels(depth) of input x.
        y_dim - int; number of data labeles.  
        z_dim - int; number of units for the latent variable z.
        generator - an callable or an object with __call__ method; for creating G network.
        discriminator - an callable or an object with __call__ method; for creating D network.
        x_reshape - a callable; for reshaping input. It is advised to rescale input between [-1, 1]
        x_scale - a callable; for rescaling input to range between [-1, 1]. 
        x_inverse_scale - callable; for reversing the scale from [-1, 1] to original input range.
        d_optimizer - optimizer for D network.
        g_optimizer - optimizer for G network.
        d_label_smooth - Desired probability for real class, to enable one side label smotiong 
                         as suggensted in http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans
        batch_size - training batch size,
        iterations - number of training iterations.
        display_step - intervals to display training stats.
        save_step - intervals to save trained model.
        oracle - If used the oracle is a callable for measuring the quality of generated samples.
                 It should be a callable or a class with __call__ function implemented.
                 the callable should take (X, reformat=False) as input an return a single float value.
        graph - The tensorflow graph to use. If None new graph is created. 
        sess - The tenserflow session to use. If None new session is created.
        sample_writer - Object of SampleWriter class.
        model_directory - model saving directory. Defaults is None.
        """
        # Data Config
        self.x_dims = x_dims
        self.x_ch = x_ch
        self.y_dim = y_dim
        self.z_size = z_dim
        self.x_reshape = x_reshape
        if x_scale is not None or x_inverse_scale is not None:
          # If one is not none the both should be not none 
          assert x_scale is not None and x_inverse_scale is not None 
        
        self.x_scale = x_scale
        self.x_inverse_scale = x_inverse_scale
        
        ######################## Generator and Discriminator Networks
        self.generator = generator
        self.discriminator = discriminator

        ######################## Training config
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_label_smooth = d_label_smooth
        self.iterations = iterations
        self.batch_size = batch_size
        self.display_step = display_step
        self.save_step = save_step
        self.sample_writer = sample_writer
        self.model_directory = model_directory
        self.oracle = oracle
        
        if graph:
            self.graph = graph
        else:
            self.graph = tf.Graph()
            
        with self.graph.as_default():
            self.build_model()
            
            if sess:
                self.sess = sess
            else:
                self.sess = tf.Session()
                
            # To save and restore checkpoints.
            self.saver = tf.train.Saver()

    def build_model(self):
        pass

    def fit(self, X, y=None, val_x=None, val_y=None):
        pass

    def _iter_stats(self, i, start_time, gLoss, dLoss, 
                    xs=None, ys=None, zs=None, ys_fake=None,
                    val_x=None, val_y=None):
        pass

    def generate(self, ys=None, n_samples=None):
        pass

    def x_reformat(self, xs):
      """ Rescale and reshape x if x_scale and x_reshape functions are provided.
      """
      if self.x_scale is not None:
        xs = self.x_scale(xs)
      if self.x_reshape is not None:
        xs = self.x_reshape(xs)
      return xs

    def _save_samples(self, i):
        if self.sample_writer is None:
            return

        n_samples = 36
        generated_x = self.generate(n_samples)
        self.sample_writer.write(generated_x, str(i))


    def _next_batch(self, x, y):
        start_index = np.random.randint(0, x.shape[0] - self.batch_size)
        return x[start_index:(start_index + self.batch_size)], \
               y[start_index:(start_index + self.batch_size)]

    def _accuracy(self, val_x, val_y, reformat=True):
        pred_y = self.predict(val_x, reformat=reformat)
        return (np.argmax(val_y, axis=1) == pred_y).mean()
        
    def predict(self, X, reformat=True):
        probs = self.predict_prob(X, reformat=reformat)
        if self.y_dim == 1:
            pred = np.zeros_like(probs)
            pred[probs > 0.5] = 1
        else:
            pred = np.argmax(probs, axis=1)
        return pred

    
    def predict_prob(self, X, reformat=True):
        self.discriminator.is_training = False
        probs_list = []
        with self.graph.as_default():
            for i in range(0, X.shape[0], self.batch_size):
                start = i
                end =  min(i+self.batch_size, X.shape[0])
                if reformat:
                    xs = self.x_reformat(X[start:end])
                else:
                    xs = X[start:end]
            
                if self.y_dim == 1:
                    probs_list.append(self.sess.run(tf.sigmoid(logits=self.Dx), feed_dict={self.real_in:xs}))
                else:
                    probs_list.append(self.sess.run(tf.nn.softmax(logits=self.Dx), feed_dict={self.real_in:xs}))
                
        self.discriminator.is_training = True
        return np.vstack(probs_list)
        
    def save_model(self, model_file_name):
      if self.model_directory is None:
        return 'ERROR: Model directory is None'
      if not os.path.exists(self.model_directory):
          os.makedirs(self.model_directory)
      return self.saver.save(self.sess, os.path.join(self.model_directory, model_file_name))
    
    def restore_model(self, model_file):
        with self.graph.as_default():
            self.saver.restore(self.sess, model_file)

    
    def generate(self, n_samples=36, ys=None):
      """ Generate samples.
      
        :param n_samples: number of samples to generate if ys is not specified. 
      """ 
      if ys is not None:
          n_samples = ys.shape[0]

      self.discriminator.is_training = False
      generated_x_list = []
      batch = self.batch_size
      for i in range(0, n_samples, batch):
        start = i
        end =  min(i+batch, n_samples)
        zs = np.random.uniform(-1.0,1.0,
                              size=[end-start,self.z_size]).astype(np.float32)
        if self.conditional:
          if ys is None:
            gen_ys = np.random.multinomial(1, [1.0 / float(self.y_dim+1)]*(self.y_dim+1), end-start)
          else:
            gen_ys = np.concatenate((ys[start:end], np.zeros((end-start, 1))), axis=1)
          
          generated_x_list.append(self.sess.run(self.Gz, feed_dict={self.z_in:zs, 
                                                                    self.real_label:gen_ys}))
        else:                        
          generated_x_list.append(self.sess.run(self.Gz, feed_dict={self.z_in:zs}))
      
      generated_x = np.vstack(generated_x_list)
      self.discriminator.is_training = True
      return self.x_inverse_scale(generated_x) if self.x_inverse_scale is not None \
                                                else generated_x
    
     

class MultiClassGAN(BaseGAN):
    """ Implementation of Deep Convolutional Conditional Generative Adversarial Network.
    """
    def __init__(self, 
                 x_dims, x_ch, y_dim,
                 generator=None,     # Generator Net
                 discriminator=None, # Discriminator Net
                 x_reshape=None,
                 x_scale=None,
                 x_inverse_scale=None,
                 z_dim=100,
                 d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 g_loss_fn='default',
                 g_target=1.0,
                 d_label_smooth=0.75,
                 sigmoid_alpha=10,
                 l2_penalty=0.01,
                 conditional=False,
                 batch_size=128, iterations=2000,
                 display_step=100, save_step=1000,
                 oracle=None,
                 graph=None, sess=None,
                 sample_writer=None, #Directory to save sample images from generator in.
                 model_directory=None, #Directory to save trained model to.
                ):
        """
        Args:
        x_dims - list; the width of and hight of image x.
        x_ch - int; the number of channels(depth) of input x.
        y_dim - int; number of data labeles.  
        z_dim - int; number of units for the latent variable z.
        generator - an callable or an object with __call__ method; for creating G network.
        discriminator - an callable or an object with __call__ method; for creating D network.
        x_reshape - a callable; for reshaping input. It is advised to rescale input between [-1, 1]
        x_scale - a callable; for rescaling input to range between [-1, 1]. 
        x_inverse_scale - callable; for reversing the scale from [-1, 1] to original input range.
        d_optimizer - optimizer for D network.
        g_optimizer - optimizer for G network.
        g_loss_fn - type of loss function used for G. Options include:
                    ['default', 'smoothed', 'sigmoid', 'feature_matching', 
                     'feature_default', 'l2_default', 'least_square']
        g_target - the target probability when g_loss_fn='smoothed'.
                    For Generated instances. For G smoting set to value < 1.0.
        d_label_smooth - Desired probability for real class, to enable one side label smotiong 
                         as suggensted in http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans
        sigmoid_alpha - alpha values when g_loss_fn='sigmoid'
        l2_penalty - l2 penalty coefficient when g_loss_fn='l2_default'
        batch_size - training batch size,
        iterations - number of training iterations.
        display_step - intervals to display training stats.
        save_step - intervals to save trained model.
        oracle - If used the oracle is a callable for measuring the quality of generated samples.
                 It should be a callable or a class with __call__ function implemented.
                 the callable should take (X, reformat=False) as input an return a single float value.
        graph - The tensorflow graph to use. If None new graph is created. 
        sess - The tenserflow session to use. If None new session is created.
        sample_writer - Object of SampleWriter class.
        model_directory - model saving directory. Defaults is None.
        """
        ######################## Training config
        assert g_loss_fn in ['default', 'smoothed', 'sigmoid', 'feature_matching', 
                             'feature_default', 'l2_default', 'least_square']
        self.g_loss_fn = g_loss_fn
        if self.g_loss_fn == 'feature_matching' or self.g_loss_fn == 'feature_default':
            assert matching_layer == -1 or matching_layer < len(conv_units)
            self.matching_layer = matching_layer if matching_layer != -1 else len(conv_units) - 1
        self.sigmoid_alpha = sigmoid_alpha
        self.g_target = g_target
        self.l2_penalty = l2_penalty
        self.conditional = conditional

        super(MultiClassGAN, self).__init__(
            x_dims, x_ch, y_dim, generator=generator, discriminator=discriminator, z_dim=z_dim,
            x_reshape=x_reshape, x_scale=x_scale, x_inverse_scale=x_inverse_scale,
            d_optimizer=d_optimizer, g_optimizer=g_optimizer, d_label_smooth=d_label_smooth,
            batch_size=batch_size, iterations=iterations, display_step=display_step, 
            save_step=save_step, oracle=oracle, graph=graph, sess=sess, 
            sample_writer=sample_writer, model_directory=model_directory)

        
    @staticmethod
    def sigmoid_cost(input, alpha):
        exp = tf.exp(-alpha * (input - 0.5))
        return tf.divide(1.0, 1 + exp)
    
    def build_model(self):
      with self.graph.as_default():
        # Placeholders
        self.z_in = tf.placeholder(name='z_in', shape=[None,self.z_size], dtype=tf.float32) #Random vector
        self.real_in = tf.placeholder(name='real_in',
          shape=[None] + self.x_dims + [self.x_ch], dtype=tf.float32) #Real images
        self.real_label = tf.placeholder(name='real_label', 
          shape=[None, self.y_dim + 1], dtype=tf.float32) #real image labels
        self.fake_label = tf.placeholder(name='fake_label', 
          shape=[None, self.y_dim + 1], dtype=tf.float32) #fake image labels
        # One side D label smoothing
        self.real_label = self.real_label * self.d_label_smooth

        self.Gz = self.generator(self.z_in, ys=self.real_label if self.conditional else None) # Condition generator on real labels
        self.Dx, fm_layer_x = self.discriminator(
          self.real_in, logits=True, 
          matching_layer=self.matching_layer if self.g_loss_fn == 'feature_matching' else None)
        self.Dg, fm_layer_g = self.discriminator(
          self.Gz, reuse=True, logits=True,
          matching_layer=self.matching_layer if self.g_loss_fn == 'feature_matching' else None)

        Dx_softmax = tf.nn.softmax(logits=self.Dx)
        Dg_softmax = tf.nn.softmax(logits=self.Dg)
        
        # d_loss and g_loss together define the optimization objective of the GAN.
        
        if self.g_loss_fn == 'least_square':
            ls_dx = 0.5 * tf.reduce_mean(tf.square(tf.subtract(Dx_softmax, self.real_label)))
            ls_dg = 0.5 * tf.reduce_mean(tf.square(tf.subtract(Dg_softmax, self.fake_label)))
            self.d_loss = ls_dx + ls_dg
        else:
            d_loss_real = tf.nn.softmax_cross_entropy_with_logits(logits=self.Dx, 
                                                                  labels=self.real_label)
            d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=self.Dg, 
                                                                  labels=self.fake_label)
            self.d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]
                    
        if self.g_loss_fn == 'smoothed':
          self.g_loss = -tf.reduce_mean(
              (1 - self.g_target) * tf.log(Dg_softmax[:, -1]) + 
              self.g_target * tf.log(1. - Dg_softmax[:, -1])
          )
        elif self.g_loss_fn == 'sigmoid':
          self.g_loss = -tf.reduce_mean(tf.log(1 - MultiClassGAN.sigmoid_cost(
            Dg_softmax[:, -1], self.sigmoid_alpha)))
        elif self.g_loss_fn == 'feature_matching':
          self.g_loss = tf.reduce_mean(tf.square(tf.subtract(fm_layer_x, fm_layer_g)))
        elif self.g_loss_fn == 'feature_default':
          self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1])) + \
                        tf.reduce_mean(tf.square(tf.subtract(fm_layer_x, fm_layer_g)))
        elif self.g_loss_fn == 'l2_default':
          g_l2_loss = 0.
          for w in g_vars:
              g_l2_loss += (self.l2_penalty * tf.reduce_mean(tf.nn.l2_loss(w)))
          self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1])) + g_l2_loss
        elif self.g_loss_fn == 'least_square': # based on https://arxiv.org/abs/1611.04076
          self.g_loss = 0.5 * tf.reduce_mean(tf.square((1. - Dg_softmax[:, -1]) - 1))
        else:
          self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1]))
                    
        # Compute gradients
        trainerD = self.d_optimizer
        trainerG = self.g_optimizer
        d_grads = trainerD.compute_gradients(self.d_loss, d_vars) #Only update the weights for the discriminator network.
        g_grads = trainerG.compute_gradients(self.g_loss, g_vars) #Only update the weights for the generator network.

        ## For Debuging
        d_grads_decomposed, _ = list(zip(*d_grads))
        g_grads_decomposed, _ = list(zip(*g_grads))
        self.d_grad_norm = tf.global_norm(d_grads_decomposed)
        self.g_grad_norm = tf.global_norm(g_grads_decomposed)
        self.d_w_norm = tf.global_norm(d_vars)
        self.g_w_norm = tf.global_norm(g_vars)
        ##
    
        self.update_D = trainerD.apply_gradients(d_grads)
        self.update_G = trainerG.apply_gradients(g_grads)
    
    def _iter_stats(self, i, start_time, gLoss, dLoss, 
                    xs=None, ys=None, zs=None, ys_fake=None,
                    val_x=None, val_y=None):
      d_grad_norm, g_grad_norm, oracle_x, d_w_norm, g_w_norm = self.sess.run(
        (self.d_grad_norm, self.g_grad_norm, self.Gz, self.d_w_norm, self.g_w_norm),
        feed_dict={self.z_in:zs, self.real_in:xs, 
                   self.real_label:ys, self.fake_label:ys_fake})
      
      tr_acc = None
      if xs is not None and ys is not None and ys_fake is not None:
        tr_x = np.concatenate((xs, oracle_x), axis=0)
        tr_y = np.concatenate((ys, ys_fake), axis=0)
        tr_acc = self._accuracy(tr_x, tr_y, reformat=False) 
      
      v_acc = None
      if val_x is not None and val_y is not None:
          v_acc = self._accuracy(val_x, val_y) 
      
      oracle_acc = None
      if self.oracle is not None:
          oracle_acc = self.oracle(oracle_x)
          
      if i == 0:
          print('{0:5}| {1:6}| {2:5}| {3:4}| {4:6}| {5:6}| {6:6}| {7:5}| {8:4}| {9:6}| {10:6}'.format(
              'i', 'GLOSS', 'DLOSS', 'TIME', 'GGRAD', 'DGRAD', 'TR_ACC','V_ACC', 'ORA', 'DW', 'GW'))
      
      print('{0:5}| {1:5.3}| {2:5.3}| {3:4}s| {4}| {5}| {6}| {7}| {8}| {9}| {10}'.format(
          i, gLoss, dLoss, int(time.time()-start_time), 
          '      ' if g_grad_norm is None else '{:6.4}'.format(g_grad_norm),
          '      ' if d_grad_norm is None else '{:6.4}'.format(d_grad_norm),
          '      ' if tr_acc is None else '{:6.3}'.format(tr_acc),
          '     ' if v_acc is None else '{:5.3}'.format(v_acc),
          '    ' if oracle_acc is None else '{:4.2}'.format(oracle_acc),
          '      ' if d_w_norm is None else '{:6.4}'.format(d_w_norm),
          '      ' if g_w_norm is None else '{:6.4}'.format(g_w_norm)))
  
    def fit(self, X, y=None, val_x=None, val_y=None):
        start = time.time()
        self.discriminator.is_training = True
        with self.graph.as_default():  
            self.sess.run(tf.global_variables_initializer())
            for i in range(self.iterations):
                zs = np.random.uniform(-1.0, 1.0,
                                       size=[self.batch_size, self.z_size]).astype(np.float32)
                xs, ys = self._next_batch(X, y)
                xs = self.x_reformat(xs)

                # Create space for the fake class label for the real data labels
                ys = np.concatenate((ys, np.zeros_like(ys[:,0])[:,None]), axis=1)
                # Create the labels for the generated data.
                ys_fake = np.zeros_like(ys)
                ys_fake[:,-1] = 1

                _, dLoss = self.sess.run(
                    [self.update_D, self.d_loss],
                    feed_dict={self.z_in:zs, self.real_in:xs, 
                               self.real_label:ys, self.fake_label:ys_fake})
                _, gLoss = self.sess.run(
                    [self.update_G, self.g_loss],
                    feed_dict={self.z_in:zs, self.real_in:xs, self.real_label:ys})
                    
                if i % self.display_step == 0:
                    self._iter_stats(i, start, gLoss, dLoss, 
                                     xs=xs, ys=ys, zs=zs, ys_fake=ys_fake,
                                     val_x=val_x, val_y=val_y)
                    self._save_samples(i)
                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                  self.save_model('model-'+str(i)+'.cptk')
                  print("Saved Model")
    
            self._iter_stats(i, start, gLoss, dLoss, 
                              xs=xs, ys=ys, zs=zs, ys_fake=ys_fake,
                              val_x=val_x, val_y=val_y)
            self._save_samples(i)
            if self.model_directory is not None:
              self.save_model('model-'+str(i)+'.cptk')
              print("Saved Model")
        self.discriminator.is_training = False
        
        
class FlatGAN(BaseGAN):
    """ Implementation of Deep Convolutional Conditional Generative Adversarial Network.
    """
    def __init__(self, 
                 x_dims, x_ch, y_dim,
                 generator=None,     # Generator Net
                 discriminator=None, # Discriminator Net
                 x_reshape=None,
                 x_scale=None,
                 x_inverse_scale=None,
                 z_dim=100,
                 d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                 d_label_smooth=1.,
                 batch_size=128, iterations=2000,
                 display_step=100, save_step=1000,
                 d_iter=1,  # number of discriminator update for each generator update
                 conditional=False,
                 oracle=None,
                 graph=None, sess=None,
                 sample_writer=None, #Directory to save sample images from generator in.
                 model_directory=None, #Directory to save trained model to.
                ):
        
        ######################## Training config
        self.d_iter = d_iter
        self.conditional = conditional

        super(FlatGAN, self).__init__(
            x_dims, x_ch, y_dim, generator=generator, discriminator=discriminator, z_dim=z_dim,
            x_reshape=x_reshape, x_scale=x_scale, x_inverse_scale=x_inverse_scale,
            d_optimizer=d_optimizer, g_optimizer=g_optimizer, d_label_smooth=d_label_smooth,
            batch_size=batch_size, iterations=iterations, display_step=display_step, 
            save_step=save_step, oracle=oracle, graph=graph, sess=sess, 
            sample_writer=sample_writer, model_directory=model_directory)
        
    def build_model(self):
        with self.graph.as_default():
            n_features = 1
            for dim in self.x_dims:
                n_features *= dim

            n_features *= self.x_ch
            # Placeholders
            self.z_in = tf.placeholder(shape=[None,self.z_size], dtype=tf.float32)
            self.real_in = tf.placeholder(
              shape=[None, n_features], dtype=tf.float32) #Real samples
            self.real_label = tf.placeholder(
              shape=[None, self.y_dim + (0 if self.y_dim == 1 else 1)], dtype=tf.float32) #real sample labels
            self.fake_label = tf.placeholder(
              shape=[None, self.y_dim + (0 if self.y_dim == 1 else 1)], dtype=tf.float32) #fake sample labels
            # One side D label smoothing
            self.real_label = self.real_label * self.d_label_smooth
            
            # Condition generator on real labels
            self.Gz = self.generator(self.z_in, ys=self.real_label if self.conditional else None) 
            self.Dx, _ = self.discriminator(self.real_in, logits=True)
            self.Dg, _ = self.discriminator(self.Gz, reuse=True, logits=True)

            if self.y_dim == 1:
                Dg_softmax = tf.sigmoid(self.Dg)
            else:
                Dg_softmax = tf.nn.softmax(logits=self.Dg)

                
            # D Loss
            d_loss_real = tf.nn.softmax_cross_entropy_with_logits(logits=self.Dx, 
                                                                  labels=self.real_label)
            d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=self.Dg, 
                                                                  labels=self.fake_label)
            self.d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
                
            # G Loss
            if self.y_dim == 1:
                self.g_loss = -tf.reduce_mean(tf.log(Dg_softmax))
            else:
                self.g_loss = -tf.reduce_mean(tf.log(1. - Dg_softmax[:, -1]))

            tvars = tf.trainable_variables()
            d_vars = [var for var in tvars if 'd_' in var.name]
            g_vars = [var for var in tvars if 'g_' in var.name]
            
            # Compute gradients
            trainerD = self.d_optimizer
            trainerG = self.g_optimizer
            d_grads = trainerD.compute_gradients(self.d_loss, d_vars) #Only update the weights for the discriminator network.
            g_grads = trainerG.compute_gradients(self.g_loss, g_vars) #Only update the weights for the generator network.

            ## For Debuging
            d_grads_decomposed, _ = list(zip(*d_grads))
            g_grads_decomposed, _ = list(zip(*g_grads))
            self.d_grad_norm = tf.global_norm(d_grads_decomposed)
            self.g_grad_norm = tf.global_norm(g_grads_decomposed)
            self.d_w_norm = tf.global_norm(d_vars)
            self.g_w_norm = tf.global_norm(g_vars)
            ##

            self.update_D = trainerD.apply_gradients(d_grads)
            self.update_G = trainerG.apply_gradients(g_grads)
    
    def _iter_stats(self, i, start_time, gLoss, dLoss, 
                    xs=None, ys=None, zs=None, ys_fake=None,
                    val_x=None, val_y=None):
      d_grad_norm, g_grad_norm, oracle_x, d_w_norm, g_w_norm = self.sess.run(
        (self.d_grad_norm, self.g_grad_norm, self.Gz, self.d_w_norm, self.g_w_norm),
        feed_dict={self.z_in:zs, self.real_in:xs, 
                   self.real_label:ys, self.fake_label:ys_fake})
      
      tr_acc = None
      if xs is not None and ys is not None and ys_fake is not None:
        tr_x = np.concatenate((xs, oracle_x), axis=0)
        tr_y = np.concatenate((ys, ys_fake), axis=0)
        tr_acc = self._accuracy(tr_x, tr_y, reformat=False) 
      
      v_acc = None
      if val_x is not None and val_y is not None:
          v_acc = self._accuracy(val_x, val_y) 
      
      oracle_acc = None
      if self.oracle is not None:
          oracle_acc = self.oracle(oracle_x)
          
      if i == 0:
          print ('{0:5}| {1:6}| {2:5}| {3:4}| {4:6}| {5:6}| {6:6}| {7:5}| {8:4}| {9:6}| {10:6}'.format(
              'i', 'GLOSS', 'DLOSS', 'TIME', 'GGRAD', 'DGRAD', 'TR_ACC','V_ACC', 'ORA', 'DW', 'GW'))
                
      print('{0:5}| {1:5.3}| {2:5.3}| {3:4}s| {4}| {5}| {6}| {7}| {8}| {9}| {10}'.format(
          i, gLoss, dLoss, int(time.time()-start_time), 
          '      ' if g_grad_norm is None else '{:6.4}'.format(g_grad_norm),
          '      ' if d_grad_norm is None else '{:6.4}'.format(d_grad_norm),
          '      ' if tr_acc is None else '{:6.3}'.format(tr_acc),
          '     ' if v_acc is None else '{:5.3}'.format(v_acc),
          '    ' if oracle_acc is None else '{:4.2}'.format(oracle_acc),
          '      ' if d_w_norm is None else '{:6.4}'.format(d_w_norm),
          '      ' if g_w_norm is None else '{:6.4}'.format(g_w_norm)))
  
    def fit(self, X, y=None, val_x=None, val_y=None):
        start = time.time()
        self.discriminator.is_training = True
        with self.graph.as_default():  
            self.sess.run(tf.global_variables_initializer())
            for i in range(self.iterations):
                for j in range(self.d_iter):
                    zs = np.random.uniform(-1.0, 1.0,
                                           size=[self.batch_size, self.z_size]).astype(np.float32)
                    xs, ys = self._next_batch(X, y)
                    xs = self.x_reformat(xs)

                    if self.y_dim != 1:
                        # Create space for the fake class label for the real data labels
                        ys = np.concatenate((ys, np.zeros_like(ys[:,0])[:,None]), axis=1)

                    # Create the labels for the generated data.
                    ys_fake = np.zeros_like(ys)
                    if self.y_dim != 1:
                        ys_fake[:,-1] = 1

                    _, dLoss = self.sess.run(
                        [self.update_D, self.d_loss],
                        feed_dict={self.z_in:zs, self.real_in:xs, 
                                   self.real_label:ys, self.fake_label:ys_fake})
                
                _, gLoss = self.sess.run(
                    [self.update_G, self.g_loss],
                    feed_dict={self.z_in:zs, self.real_in:xs, self.real_label:ys})
                    
                if i % self.display_step == 0:
                    self._iter_stats(i, start, gLoss, dLoss, 
                                     xs=xs, ys=ys, zs=zs, ys_fake=ys_fake,
                                     val_x=val_x, val_y=val_y)
                    self._save_samples(i)
                if i % self.save_step == 0 and i != 0 and self.model_directory is not None:
                  self.save_model('model-'+str(i)+'.cptk')
                  print("Saved Model")
    
            self._iter_stats(i, start, gLoss, dLoss, 
                              xs=xs, ys=ys, zs=zs, ys_fake=ys_fake,
                              val_x=val_x, val_y=val_y)
            self._save_samples(i)
            if self.model_directory is not None:
              self.save_model('model-'+str(i)+'.cptk')
              print("Saved Model")
        self.discriminator.is_training = False
