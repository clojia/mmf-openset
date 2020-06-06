import sys
import os.path
sys.path.insert(0, os.path.abspath("../"))

from generative.vae import VariationalAutoencoder
from util.sample_writer import ImageGridWriter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../data/MNIST_data/", 
                                  one_hot=True)

print mnist.train.images.shape
print mnist.train.labels.shape

cvae_open = VariationalAutoencoder(
    28*28, y_dim=10, 
    z_dim=256, h_dims=[256, 128],
    optimizer=tf.train.AdamOptimizer(),
    batch_size=128, 
    iterations=20,
    display_step=10,
    save_step=500,
    # model_directory='../../data/models/cvae'
)

cvae_open.fit(mnist.train.images, mnist.train.labels)

gen_xs = cvae_open.generate(n_samples=36)
# gen_xs = cvae_open.generate(ys=ys_gen)
gen_imgs = ImageGridWriter.merge_img(np.reshape(gen_xs[0:len(gen_xs)],[len(gen_xs), 28, 28]))
plt.imshow(gen_imgs, cmap='gray')
plt.show()