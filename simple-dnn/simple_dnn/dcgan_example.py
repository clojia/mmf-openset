# ---------------------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from generative.discriminator import DiscriminatorDC
from generative.gan import MultiClassGAN
from generative.generator import GeneratorDC
from util.format import UnitPosNegScale, reshape_pad
from util.sample_writer import ImageGridWriter

mnist = input_data.read_data_sets("../../data/MNIST_data/", 
                                  one_hot=True)

print mnist.train.images.shape
print mnist.train.labels.shape

discriminator = DiscriminatorDC(10,  # y_dim
                              [16,32,64], # conv_units
                              hidden_units=None,
                              kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
                              d_activation_fn=tf.contrib.keras.layers.LeakyReLU,
                              f_activation_fns=tf.nn.relu,
                              dropout=False, keep_prob=0.5)
generator = GeneratorDC([32, 32],#x_dims
                        1, # x_ch
                        [64,32,16], # g_conv_units
                        g_kernel_sizes=[5,5], g_strides=[2, 2], g_paddings='SAME',
                        g_activation_fn=tf.nn.relu)



dcgan = MultiClassGAN([32, 32], # x_dim 
                      1, # x_ch 
                      10, # y_dim 
                      z_dim=100,
                      generator=generator,     # Generator Net
                      discriminator=discriminator, # Discriminator Net
                      x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                      x_scale=UnitPosNegScale.scale,
                      x_inverse_scale=UnitPosNegScale.inverse_scale,
                      d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                      g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                      g_loss_fn='default',
                      d_label_smooth=0.75,
                      ## Training config
                      batch_size=128,
                      iterations=5,
                      display_step=1,
                      save_step=500,
                      sample_writer= ImageGridWriter('../../data/figs/closed', grid_size=[6, 6], 
                                                      img_dims=[32, 32])
            #           model_directory='../data/models/closed'
                      )

dcgan.fit(mnist.train.images, mnist.train.labels,
          val_x=mnist.validation.images, val_y=mnist.validation.labels)

n_samples = 36
# ys_gen = np.zeros([n_samples, mnist.train.labels.shape[1] + 1])
# ys_gen[:, np.random.randint(0, mnist.train.labels.shape[1], size=n_samples)] = 1
        
gen_xs = dcgan.generate(n_samples)
gen_imgs = ImageGridWriter.merge_img(np.reshape(gen_xs[0:n_samples],[n_samples, dcgan.x_dims[0], dcgan.x_dims[1]]))
plt.imshow(gen_imgs, cmap='gray')
plt.show()
