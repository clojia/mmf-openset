import argparse
import logging

import sys
import os.path

import tensorflow as tf
import time
import datetime

sys.path.insert(0, os.path.abspath("./simple-dnn"))

from g_openmax import GOpenmax
from open_net import OpenNetFlat, OpenNetCNN
from openmax import OpenMaxFlat, OpenMaxCNN
from central_opennet import CentralOpennetFlat, CentralOpennetCNN
from exp_opennet_util import load_open_dataset, save_pickle_gz
from simple_dnn.generative.discriminator import DiscriminatorDC, DiscriminatorFlat
from simple_dnn.generative.gan import MultiClassGAN, FlatGAN
from simple_dnn.generative.generator import GeneratorDC, GeneratorFlat
from simple_dnn.util.sample_writer import ImageGridWriter
from simple_dnn.util.format import UnitPosNegScale, reshape_pad, flatten_unpad
from util.openworld_sim import OpenWorldSim, OpenWorldMsData

# Open Models
def get_flat_model_factories(model_name, dataset_name, open_dataset, z_dim=None):
    if dataset_name == 'ms':
        if model_name == 'mmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[256, 64],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn='euclidean',
                batch_size=128,
                iterations=10000,
                display_step=-1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                contamination=0.01,
            )
        elif model_name == 'ii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[256, 64],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn='euclidean',
                batch_size=128,
                iterations=10000,
                display_step=-1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                contamination=0.01,
            )
        elif model_name == 'ce':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[256, 64],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn='euclidean',
                batch_size=128,
                iterations=10000,
                display_step=-1000,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                contamination=0.01,
            )
        elif model_name == 'ceii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[256, 64],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn='euclidean',
                batch_size=128,
                iterations=10000,
                display_step=-1000,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                contamination=0.01,)

        elif model_name == 'central':
            return lambda : CentralOpennetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                h_dims=[256, 64],
                dropout = True, keep_prob=0.25,
                decision_dist_fn='euclidean',
                batch_size=128,
                iterations=10000,
                display_step=-1000,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                contamination=0.01,
                penalty=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                h_dims=[256, 64],
                dropout = True, keep_prob=0.25,
                decision_dist_fn='eucos',
                batch_size=128,
                iterations=10000,
                display_step=-1000,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                tailsize = 2,
                alpharank = 4,
            )
    elif dataset_name == 'android':
        if model_name == 'triplet':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='triplet_center_loss',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=True,
                contamination=0.01, margin=8.0
            )
        elif model_name == 'tripletmmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='triplet_center_loss',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=12000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=True, tc_loss=True,
                contamination=0.01, mmf_comb=0.4, margin=8.0
            )
        elif model_name == 'ceiimmf':
            return lambda: OpenNetFlat(
                open_dataset.train_data().shape[1],  # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu,  # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout=True, keep_prob=0.9,
                dist='mean_separation_spread',  # 'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn='euclidean',  # 'mahalanobis',
                batch_size=256,
                iterations=12000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.6
            )
        elif model_name == 'iimmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=45000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.4
            )
        elif model_name == 'ii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=False,
                contamination=0.01,
            )
        elif model_name == 'ce':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='others',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'cemmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='others',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=1.0
            )
        elif model_name == 'ceii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'central':
            return lambda : CentralOpennetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                contamination=0.01,
                penalty=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                decision_dist_fn = 'eucos',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                tailsize = 5,
                alpharank = 3,
            )
        elif model_name == 'g_openmax':
            return lambda : GOpenmax(
                gan_factory = lambda y_dim: FlatGAN(
                        [open_dataset.train_data().shape[1]], # x_dim 
                        1, # x_ch 
                        y_dim, # y_dim
                        z_dim=100,
                        generator=GeneratorFlat(
                            [open_dataset.train_data().shape[1]],#x_dims
                            1, # x_ch
                            [128, 256], #[1200,1200], # hideen_units
                            g_activation_fn=tf.nn.relu),     # Generator Net
                        discriminator=DiscriminatorFlat(
                            6,  # y_dim
                            hidden_units=[256, 128,64], #[240, 240],#
                            f_activation_fns=lambda net:tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(net), 
                            dropout=False, keep_prob=0.5), # Discriminator Net
                        x_reshape=None,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        d_label_smooth=1.,
                        conditional=True,
                        ## Training config
                        batch_size=128,
                        iterations=5000,
                        display_step=-500,
                        save_step=500,
                        ), 
                openmax_factory= lambda y_dim: OpenMaxFlat(
                        open_dataset.train_data().shape[1], # x_dim
                        y_dim=y_dim,
                        h_dims=[64],
                        activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                        dropout = True, keep_prob=0.9,
                        decision_dist_fn = 'eucos',#'mahalanobis',
                        batch_size=256,
                        iterations=10000,
                        display_step=-500,
                        save_step=1000,
                        tailsize = 5,
                        alpharank = 3,
                        ), 
                classifier_factory = lambda y_dim: OpenMaxFlat(
                        open_dataset.train_data().shape[1], # x_dim
                        y_dim=y_dim,
                        h_dims=[64],
                        activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                        dropout = True, keep_prob=0.9,
                        decision_dist_fn = 'eucos',#'mahalanobis',
                        batch_size=256,
                        iterations=10000,
                        display_step=-500,
                        save_step=1000,
                        tailsize = 5,
                        alpharank = 3,
                        ), 
                y_dim=6, batch_size=128,
                unpad_flatten=None)

    elif dataset_name == 'mnist':
        if model_name == 'mmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=20000,
                display_step=-1,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=20000,
                display_step=-1,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ce':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ceiimmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=1.0,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'ceii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                z_dim=6 if z_dim is None else z_dim,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                dist='mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'mahalanobis',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=True, div_loss=False,
                contamination=0.01,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=6,
                h_dims=[512, 128],
                dropout = True, keep_prob=0.25,
                decision_dist_fn = 'eucos',
                batch_size=128,
                iterations=10000,
                display_step=-1,
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                tailsize = 20,
                alpharank = 4,
            )

def get_cnn_model_factories(model_name, dataset_name, open_dataset, z_dim=None):
    if dataset_name == 'msadjmat':
        if model_name == 'triplet':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='triplet_center_loss',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=True,
                contamination=0.01, margin=2.0
                )
        if model_name == 'tripletmmf':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='triplet_center_loss',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=True, tc_loss=True,
                contamination=0.01, mmf_comb=0.4, margin=2.0
                )
        if model_name == 'cemmf':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=1.0
                )

        if model_name == 'iimmf':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.5
                )

        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
                )
        elif model_name == 'ce':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
                )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01
                )
        elif model_name == 'ceiimmf':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=1.0
                )
        elif model_name == 'central':
            return lambda : CentralOpennetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                contamination=0.01,
                penalty=0.01,
                )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                decision_dist_fn='eucos',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 3,
                )
        elif model_name == 'g_openmax':
            return lambda : GOpenmax(
                gan_factory = lambda y_dim: FlatGAN(
                        [open_dataset.train_data().shape[1]], # x_dim 
                        1, # x_ch 
                        y_dim, # y_dim
                        z_dim=100,
                        generator=GeneratorFlat(
                            [open_dataset.train_data().shape[1]],#x_dims
                            1, # x_ch
                            [128, 256], #[1200,1200], # hideen_units
                            g_activation_fn=tf.nn.relu),     # Generator Net
                        discriminator=DiscriminatorFlat(
                            6,  # y_dim
                            hidden_units=[256, 128,64], #[240, 240],#
                            f_activation_fns=lambda net:tf.contrib.keras.layers.LeakyReLU(alpha=0.2)(net), 
                            dropout=False, keep_prob=0.5), # Discriminator Net
                        x_reshape=None,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        d_label_smooth=1.,
                        conditional=True,
                        ## Training config
                        batch_size=128,
                        iterations=5000,
                        display_step=-500,
                        save_step=500,
                        ), 
                openmax_factory= lambda y_dim: OpenMaxCNN(
                            [67, 67],  # x_dim
                            1,  #x_ch
                            y_dim,  #y_dim
                            [32, 64], # conv_units,
                            [256],      #hidden_units
                            kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                            pooling_enable=True, pooling_kernel=[3,3],
                            pooling_stride=[2,2], pooling_padding='SAME',
                            pooling_type='max',
                            activation_fn=tf.nn.relu,

                            x_scale=UnitPosNegScale.scale,
                            x_inverse_scale=UnitPosNegScale.inverse_scale,
                            x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                            c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                            decision_dist_fn='eucos',#
                            dropout = True, keep_prob=0.9,
                            batch_size=256,
                            iterations=5000,
                            display_step=-500,
                            save_step=500,
                            model_directory=None,  # Directory to save trained model to.
                            tailsize = 20,
                            alpharank = 3,
                            ),
                classifier_factory = lambda y_dim: OpenNetFlat(
                            open_dataset.train_data().shape[1], # x_dim
                            y_dim=y_dim,
                            z_dim=y_dim,
                            h_dims=[64],
                            dropout=True, keep_prob=0.9,
                            dist='mean_separation_spread',#'class_mean',#
                            decision_dist_fn='euclidean',
                            batch_size=256,
                            iterations=10000,
                            display_step=-1000,
                            ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                            activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                            x_scale=UnitPosNegScale.scale,
                            x_inverse_scale=UnitPosNegScale.inverse_scale,
                            opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                            recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                            c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                            contamination=0.01,),
                y_dim=6, batch_size=128,
                unpad_flatten=None)

    elif dataset_name == 'android':
        if model_name == 'mmf':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01,  mmf_comb=0
            )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([37,37], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.9,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,  mmf_comb=0
            )

    elif dataset_name == 'mnist':
        if model_name == 'triplet':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_chs
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=True,
                contamination=0.01, margin=2.0
            )
        elif model_name == 'tripletmmf':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='triplet_center_loss',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=True, tc_loss=True,
                contamination=0.01, margin=2.0, mmf_comb=0.5
            )
        elif model_name == 'ceiimmf':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=1.0
            )
        elif model_name == 'iimmf':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.2
            )

        elif model_name == 'cemmf':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='others',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=True, tc_loss=False,
                contamination=0.01, mmf_comb=1.0
            )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,  mmf_extension=False,
                contamination=0.01
            )
        elif model_name == 'ce':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=6 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'central':
            return lambda : CentralOpennetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                contamination=0.01,
                penalty=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                6,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                decision_dist_fn='eucos',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 3,
            )
        elif model_name == 'g_openmax':
            return lambda : GOpenmax(
                gan_factory = lambda y_dim: MultiClassGAN(
                        [32, 32], # x_dim 
                        1, # x_ch 
                        y_dim, # y_dim 
                        z_dim=100,
                        generator=GeneratorDC([32, 32],#x_dims
                        1, # x_ch
                        [64,32,16], # g_conv_units
                        g_kernel_sizes=[5,5], g_strides=[2, 2], g_paddings='SAME',
                        g_activation_fn=tf.nn.relu),     # Generator Net
                        discriminator=DiscriminatorDC(6,  # y_dim
                                [16,32,64], # conv_units
                                hidden_units=None,
                                kernel_sizes=[5,5], strides=[2, 2], paddings='SAME',
                                d_activation_fn=tf.contrib.keras.layers.LeakyReLU,
                                f_activation_fns=tf.nn.relu,
                                dropout=False, keep_prob=0.5), # Discriminator Net
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        d_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_optimizer=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5),
                        g_loss_fn='default',
                        d_label_smooth=0.75,
                        ## Training config
                        batch_size=128,
                        iterations=5000,
                        display_step=-500,
                        save_step=500,
                        ), 
                openmax_factory= lambda y_dim: OpenMaxCNN(
                        [32, 32],  1,  #x_ch
                        y_dim,  #y_dim
                        [32, 64], # conv_units,
                        [256, 128],      #hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3,3],
                        pooling_stride=[2,2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout = True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        save_step=500,
                        tailsize = 20,
                        alpharank = 3), 
                classifier_factory = lambda y_dim: OpenMaxCNN(
                        [32, 32],  1,  #x_ch
                        y_dim,  #y_dim
                        [32, 64], # conv_units,
                        [256, 128],      #hidden_units
                        kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                        pooling_enable=True, pooling_kernel=[3,3],
                        pooling_stride=[2,2], pooling_padding='SAME',
                        pooling_type='max',
                        activation_fn=tf.nn.relu,
                        x_scale=UnitPosNegScale.scale,
                        x_inverse_scale=UnitPosNegScale.inverse_scale,
                        x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),
                        c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                        decision_dist_fn='eucos',
                        dropout = True, keep_prob=0.2,
                        batch_size=256,
                        iterations=1500,
                        display_step=-500,
                        tailsize = 20,
                        alpharank = 3), 
                y_dim=6, batch_size=128,
                unpad_flatten=flatten_unpad([32,32], [28,28],1))

# Closed Models
def get_closed_flat_model_factories(model_name, dataset_name, open_dataset, z_dim=None):
    if dataset_name == 'android':
        if model_name == 'ce':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=8,
                z_dim=8 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='others',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'cemmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=8,
                z_dim=8 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='others',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.5
            )

        elif model_name == 'ii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                z_dim=9 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'iimmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                z_dim=9 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.3
            )
        elif model_name == 'ceii':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                z_dim=9 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'ceiimmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                z_dim=9 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='mean_separation_spread',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.5
            )
        elif model_name == 'triplet':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                z_dim=9 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='triplet_center_loss',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=False, intra_loss=False, div_loss=False, mmf_extension=False, tc_loss=True,
                contamination=0.01, margin=0.5
            )
        elif model_name == 'tripletmmf':
            return lambda : OpenNetFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                z_dim=9 if z_dim is None else z_dim,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                dist='triplet_center_loss',#'min_max',#'mean_separation_spread',#'class_mean',#
                decision_dist_fn = 'euclidean',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True, tc_loss=True,
                contamination=0.01, mmf_comb=1.0
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxFlat(
                open_dataset.train_data().shape[1], # x_dim
                y_dim=9,
                h_dims=[64],
                activation_fn=tf.nn.relu, # LeakyReLUfunction, #
                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                c_opt=tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9),
                dropout = True, keep_prob=0.9,
                decision_dist_fn = 'eucos',#'mahalanobis',
                batch_size=256,
                iterations=10000,
                display_step=-500,
                save_step=1000,
                tailsize = 5,
                alpharank = 4,
            )


def get_closed_cnn_model_factories(model_name, dataset_name, open_dataset, z_dim=None):
    if dataset_name == 'msadjmat':
        if model_name == 'mmf':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                9,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=9 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.5,
                batch_size=64,
                iterations=5000,
                display_step=500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01,
                )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                9,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=9 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.5,
                batch_size=64,
                iterations=5000,
                display_step=500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
                )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                9,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                z_dim=9 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                dist='mean_separation_spread',
                decision_dist_fn='euclidean',#
                dropout = True, keep_prob=0.5,
                batch_size=64,
                iterations=8000,
                display_step=500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
                )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [67, 67],  # x_dim
                1,  #x_ch
                9,  #y_dim
                [32, 64], # conv_units,
                [256],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape= reshape_pad([63,63], [67,67], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9),

                decision_dist_fn='eucos',#
                dropout = True, keep_prob=0.5,
                batch_size=64,
                iterations=5000,
                display_step=500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 4,
                )
    elif dataset_name == 'mnist':
        if model_name == 'mmf':
            return lambda: OpenNetCNN(
                [32, 32],  # x_dim
                1,  # x_ch
                10,  # y_dim
                [32, 64],  # conv_units,
                [256, 128],  # hidden_units
                z_dim=10 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3, 3],
                pooling_stride=[2, 2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28, 28], [32, 32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout=True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False, mmf_extension=True,
                contamination=0.01, mmf_comb=0.1
            )
        elif model_name == 'ii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                10,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=10 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=False, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'ceii':
            return lambda : OpenNetCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                10,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                z_dim=10 if z_dim is None else z_dim,
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                recon_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),
                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                dist='mean_separation_spread',
                decision_dist_fn='mahalanobis',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                density_estimation_factory=None,
                ce_loss=True, recon_loss=False, inter_loss=True, intra_loss=True, div_loss=False,
                contamination=0.01,
            )
        elif model_name == 'openmax':
            return lambda : OpenMaxCNN(
                [32, 32],  # x_dim
                1,  #x_ch
                10,  #y_dim
                [32, 64], # conv_units,
                [256, 128],      #hidden_units
                kernel_sizes=[4, 4], strides=[1, 1], paddings='SAME',
                pooling_enable=True, pooling_kernel=[3,3],
                pooling_stride=[2,2], pooling_padding='SAME',
                pooling_type='max',
                activation_fn=tf.nn.relu,

                x_scale=UnitPosNegScale.scale,
                x_inverse_scale=UnitPosNegScale.inverse_scale,
                x_reshape=reshape_pad([28,28], [32,32], 1, pad=True, pad_value=-1),

                c_opt=tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5),

                decision_dist_fn='eucos',
                dropout = True, keep_prob=0.2,
                batch_size=256,
                iterations=5000,
                display_step=-500,
                save_step=500,
                model_directory=None,  # Directory to save trained model to.
                tailsize = 20,
                alpharank = 4,
            )


def single_exp(exp_id, network_type, model_name, model_factory, dataset_name, dataset, output_dir, tr_classes):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    m = model_factory()

    print('m.y_dim=', m.y_dim)
    train_start = time.time()
    m.fit(dataset.train_data(), dataset.train_label())
    train_end = time.time()

    result = {}
    result['dataset_name'] = dataset_name
    result['model_name'] = model_name
    result['network_type'] = network_type
    result['tr_classes'] = tr_classes
    try:
        result['model_config'] = m.model_config()
        result['class_mean'] = m.c_means
        result['class_cov'] = m.c_cov
        result['class_cov_inv'] = m.c_cov_inv
    except:
        pass
    result['train_decision_function'] = m.decision_function(dataset.train_data())
    result['test_decision_function'] = m.decision_function(dataset.test_data())
    try:
        result['train_dist_all_class'] = m.distance_from_all_classes(dataset.train_data())
        result['test_dist_all_class'] = m.distance_from_all_classes(dataset.test_data())
    except:
        pass
    try:
        result['test_predict_prob'] = m.predict_prob(dataset.test_data())
    except:
        pass
    try:
        result['test_predict_prob_open'] = m.predict_prob_open(dataset.test_data())
    except:
        pass
    try:
        result['test_closed_predict_y'] = m.predict(dataset.test_data())
    except:
        pass
    try:
        result['test_open_predict_y'] = m.predict_open(dataset.test_data())
    except:
        pass
    try:
        result['train_z'] = m.latent(dataset.train_data())
        result['test_z'] = m.latent(dataset.test_data())
    except:
        pass
    result['train_true_y'] = dataset.train_label()
    result['test_true_y'] = dataset.test_label()
    result['train_time'] = int(train_end - train_start)

    fmt='{dataset_name}_{network_type}_{model_name}_%Y_%m_%d_%H_%M_%S_e{exp_id}.pkl.gz'
    file_name = datetime.datetime.now().strftime(fmt).format(
        dataset_name=dataset_name, network_type=network_type, model_name=model_name, exp_id=exp_id)
    save_pickle_gz(result, os.path.join(output_dir, file_name))
    return m

def main():
    parser = argparse.ArgumentParser(description='OpenNetFlat experiments.')
    parser.add_argument('-eid', '--exp_id', required=True, dest='exp_id',
                        help='path to output directory.')
    parser.add_argument('-ds','--dataset', required=True, dest='dataset_name',
                        choices=['mnist', 'ms', 'android', 'msadjmat'], help='dataset name.')
    parser.add_argument('-n','--network', required=True, dest='network',
                        choices=['flat', 'cnn'], help='dataset name.')
    parser.add_argument('-m','--model', required=True, dest='model_name',
                        choices=['ii', 'ce', 'ceii', 'openmax', 'g_openmax', 'central', 'mmf', 'triplet', 'tripletmmf', 'cemmf', 'iimmf', 'ceiimmf'], help='model name.')
    parser.add_argument('-trc', '--tr_classes', required=True, dest='tr_classes', nargs='+',
                        type=int, help='list of training classes.')
    parser.add_argument('-o', '--outdir', required=False, dest='output_dir',
                        default='./exp_result/cnn', help='path to output directory.')
    parser.add_argument('-z', '--zdim', required=False, dest='z_dim', type=int,
                        default=None, help='[optional] dimension of z layer.')
    parser.add_argument('-s', '--seed', required=False, dest='seed', type=int,
                        default=1, help='path to output directory.')
    parser.add_argument('--closed', dest='closed', action='store_true')
    parser.add_argument('--no-closed', dest='closed', action='store_false')
    parser.set_defaults(closed=False)

    args = parser.parse_args()
    print ("Loading Dataset: " + args.dataset_name)
    open_dataset = load_open_dataset(args.dataset_name, args.tr_classes, args.seed)
    print ("Creating Model Config: " + args.network)
    if args.network == 'flat':
        if args.closed:
            model_factory = get_closed_flat_model_factories(
                args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
        else:
            model_factory = get_flat_model_factories(
                args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
    elif args.network == 'cnn':
        if args.closed:
            model_factory = get_closed_cnn_model_factories(
                args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
        else:
            model_factory = get_cnn_model_factories(
                args.model_name, args.dataset_name, open_dataset, z_dim=args.z_dim)
    print ("Starting single experiment...")
    single_exp(args.exp_id, args.network, args.model_name, model_factory,
               args.dataset_name, open_dataset, args.output_dir, args.tr_classes)
    print ("Finished single experiment...")


if __name__ == '__main__':
    main()
