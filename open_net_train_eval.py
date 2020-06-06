import sys
import os.path

#Import the libraries we will need.
from IPython.display import display
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc
import scipy
import scipy.io
import time
import pickle
import matplotlib.cm as cm
import random

from visualization import visualize_dataset_2d, visualize_dataset_nd, visualize_t_SNE

from open_net import OpenNetFlat, OpenNetCNN
from util.metrics import auc, open_set_classification_metric

from metrics import auc, open_set_classification_metric



def train_eval(disc_ae, tr_x, tr_y, val_x=None, val_y=None, val_known_mask=None,
               ts_x=None, ts_y=None, ts_known_mask=None, n_scatter=1000, unique_ys=range(7),
               train=True, plot_recon=False, grid_shape=(1,3), figsize=(12, 4), save_path=None,
               label_text_lookup=None, is_openset=True, visualize=True, acc=None):

    if train:
        disc_ae.fit(tr_x, tr_y,
                    val_x[val_known_mask] if val_x is not None else None,
                    val_y[val_known_mask, :-1] if val_y is not None else None)

    if visualize:
        z = disc_ae.latent(tr_x[:n_scatter])
        if z.shape[1] == 2:
            visualize_dataset_2d(z[:, 0], z[:, 1], np.argmax(tr_y[:n_scatter], axis=1),
                                alpha=0.5, figsize=figsize, unique_ys=unique_ys, save_path=save_path,
                                label_text_lookup=label_text_lookup)
        elif z.shape[1] == 3:
            visualize_dataset_nd(z, np.argmax(tr_y[:n_scatter], axis=1), grid_shape=(1,3), alpha=0.5,
                                loc='upper left', bbox_to_anchor=(1.04,1), figsize=figsize,
                                unique_ys=unique_ys, save_path=save_path,
                                label_text_lookup=label_text_lookup)
        else:
            visualize_dataset_nd(z, np.argmax(tr_y[:n_scatter], axis=1), grid_shape=grid_shape, alpha=0.5,
                                loc='upper left', bbox_to_anchor=(1.04,1), figsize=figsize,
                                unique_ys=unique_ys, save_path=save_path,
                                label_text_lookup=label_text_lookup)

        z = disc_ae.latent(ts_x[:n_scatter])
        if z.shape[1] == 2:
            visualize_dataset_2d(z[:, 0], z[:, 1],
                                np.argmax(ts_y[:n_scatter], axis=1), #open_mnist.test_label()[:1000, 6],
                                alpha=0.5, figsize=figsize, unique_ys=unique_ys, save_path=save_path,
                                label_text_lookup=label_text_lookup)
        elif z.shape[1] == 3:
            visualize_dataset_nd(z, np.argmax(ts_y[:n_scatter], axis=1), grid_shape=(1,3), alpha=0.5, #grid_shape=(2,3)
                                loc='upper left', bbox_to_anchor=(1.04,1), figsize=figsize,          #figsize=(12, 8)
                                unique_ys=unique_ys, save_path=save_path,
                                label_text_lookup=label_text_lookup)
        else:
            visualize_dataset_nd(z, np.argmax(ts_y[:n_scatter], axis=1), grid_shape=grid_shape, alpha=0.5,
                                loc='upper left', bbox_to_anchor=(1.04,1), figsize=figsize,
                                unique_ys=unique_ys, save_path=save_path,
                                label_text_lookup=label_text_lookup)


        ss = {'Training':{}, 'Test':{}}
        ss['Training']['avg_c_separation'], ss['Training']['min_c_separation'] = class_separation(disc_ae.c_means)
        z = disc_ae.latent(tr_x)
        ss['Training']['avg_c_spread'], ss['Training']['max_c_spread'] = class_spread(disc_ae.c_means, z, tr_y)

        if is_openset:
            ss['Test']['avg_c_separation'], ss['Test']['min_c_separation'] = class_separation(disc_ae.c_means)

            z = disc_ae.latent(ts_x[ts_known_mask])
            ss['Test']['avg_c_spread'], ss['Test']['max_c_spread'] = class_spread(disc_ae.c_means, z,
                                                                                ts_y[ts_known_mask][:, :-1])

        mean = {}
        mean['idx0'] = disc_ae.c_means[0]
        display(pd.DataFrame(ss))

    if is_openset:
        auc_score = auc(ts_y[:, -1], disc_ae.decision_function(ts_x), pos_label=1)
        print auc_score

    if plot_recon:
        gen_xs = disc_ae.reconstruct(open_mnist.train_data()[:36])
        gen_imgs = merge_img(np.reshape(gen_xs[0:len(gen_xs)],[len(gen_xs), disc_ae.x_dim[0], disc_ae.x_dim[1]]))
        plt.imshow(gen_imgs, cmap='gray')
        plt.show()

    if is_openset:
        acc = open_set_classification_metric(np.argmax(ts_y, axis=1),
                                       disc_ae.predict_open(ts_x), is_openset=is_openset, acc=acc)
    else:
        acc = open_set_classification_metric(np.argmax(ts_y, axis=1),
                                       disc_ae.predict(ts_x), is_openset=is_openset, acc=acc)

    return acc

train_eval(disc_ae_2, open_mnist.train_data(), open_mnist.train_label(),
           open_mnist.validation_data(), open_mnist.validation_label(),
           np.logical_not(open_mnist.validation_label()[:,-1].astype(bool)),
           open_mnist.test_data(), open_mnist.test_label(),
           np.logical_not(open_mnist.test_label()[:,-1].astype(bool)),
           n_scatter=1000, unique_ys=range(7)
          )



#def compare_performance(model_factory_dict, n_runs,
#                        tr_x, tr_y,
#                        val_x, val_y, val_known_mask,
#                        ts_x, ts_y, ts_known_mask, show_result=True):
#    auc_df_avg = 'auc_df_avg'
#    auc_df_std = 'auc_df_std'
#    auc_df_all = 'auc_df_all'
#    results = {key:{} for key in model_factory_dict.keys()}
#    start = time.time()
#    for model_name, factory in model_factory_dict.items():
#        print model_name,
#        auc_array = np.zeros(n_runs)
#        for n in range(n_runs):
#            model = factory()
#            model.fit(tr_x, tr_y,
#                      val_x[val_known_mask] if val_x is not None else None,
#                      val_y[val_known_mask, :-1] if val_y is not None else None)
#
#            auc_array[n] = auc(ts_y[:, -1], model.decision_function(ts_x), pos_label=1, plot=False)
#            print '{0:4}'.format(int(time.time() - start)),
#
#        print ''
#        if auc_df_avg not in results[model_name]:
#            results[model_name][auc_df_avg] = 0.
#        results[model_name][auc_df_avg] = auc_array.mean()
#
#        if auc_df_std not in results[model_name]:
#            results[model_name][auc_df_std] = 0.
#        results[model_name][auc_df_std] = auc_array.std()
#
#        results[model_name][auc_df_all] = auc_array
#
#    if show_result:
#        display(pd.DataFrame(results))

#    return results

class OpenDatasetIter:
    def __init__(self, dataset_name, dataset_factory_fn, list_tr_classes, random_seeds='auto'):
        """
        Parameters
        ----------
        dataset_name : string
            Name of the dataset.
        list_tr_classes: list of lists
            A list known classes.
        dataset_factory_fn: callable
            A function that takes a list(knwon classes) and random seed and returns open dataset.
        random_seeds: None, list of ints, 'auto'
            The random seeds to be passed to dataset_factory_fn. If None the None will be passed.
            If 'auto' then deterministic seed values ranging from 1 to len(list_tr_classes)+1 are used.
            Otherwise, if is a list then len(random_seeds) == len(list_tr_classes)
        """
        self.i = 0
        self.dataset_name = dataset_name
        self.dataset_factory_fn = dataset_factory_fn
        self.list_tr_classes = list_tr_classes
        self.n = len(self.list_tr_classes)
        if random_seeds is None:
            self.random_seeds = [None] * self.n
        elif random_seeds == 'auto':
            self.random_seeds = range(1, self.n+1)
        else:
            # number of random seeds must equal len of list_tr_classes
            assert len(random_seeds) == self.n
            self.random_seeds = random_seeds

    def __iter__(self):
        return self

    def next(self):
        if self.i < self.n:
            tr_classes = self.list_tr_classes[self.i]
            seed = self.random_seeds[self.i]
            self.i += 1
            return self.dataset_name, self.dataset_factory_fn(tr_classes, seed)
        else:
            raise StopIteration()

#OpenDatasetIter(
#        'MNIST',
#        lambda tr_classes, random_seed: OpenWorldSim(
#            mnist.train.images, mnist.train.labels,
#            val_data=mnist.validation.images, val_label=mnist.validation.labels,
#            test_data=mnist.test.images, test_label=mnist.test.labels,
#            tr_classes=tr_classes,  seed=random_seed),
#        [[0, 2, 3, 4, 6, 9], [7, 0, 9, 5, 3, 1], [9, 2, 3, 4, 0, 1]], random_seeds='auto')


def compare_performance(model_factory_dict, n_runs=1,
                        tr_x=None, tr_y=None,
                        val_x=None, val_y=None, val_known_mask=None,
                        ts_x=None, ts_y=None, ts_known_mask=None,
                        show_result=True,
                        open_dataset_iterator_factory=None):
    auc_df_avg = 'auc_df_avg'
    auc_df_std = 'auc_df_std'
    auc_df_all = 'auc_df_all'
    results = {key:{} for key in model_factory_dict.keys()}
    start = time.time()
    for model_name, factory in model_factory_dict.items():
        print model_name,
        auc_array = []
        def single_dataset_n_runs():
            for n in range(n_runs):
                model = factory()
                model.fit(tr_x, tr_y,
                          val_x[val_known_mask] if val_x is not None else None,
                          val_y[val_known_mask, :-1] if val_y is not None else None)

                auc_array.append(auc(ts_y[:, -1], model.decision_function(ts_x), pos_label=1, plot=False))
                print '{0:4}({1:4.2})'.format(int(time.time() - start), auc_array[-1]),

            print ''

        if open_dataset_iterator_factory:
            iterator = open_dataset_iterator_factory()
            for _, open_dataset in iterator:
                tr_x = open_dataset.train_data()
                tr_y = open_dataset.train_label()
                val_x = open_dataset.validation_data()
                val_y = open_dataset.validation_label()
                val_known_mask = np.logical_not(open_dataset.validation_label()[:,-1].astype(bool))
                ts_x = open_dataset.test_data()
                ts_y = open_dataset.test_label()
                ts_known_mask = np.logical_not(open_dataset.test_label()[:,-1].astype(bool))

                single_dataset_n_runs()
        else:
            single_dataset_n_runs()

        auc_array = np.array(auc_array)
        if auc_df_avg not in results[model_name]:
            results[model_name][auc_df_avg] = 0.
        results[model_name][auc_df_avg] = auc_array.mean()

        if auc_df_std not in results[model_name]:
            results[model_name][auc_df_std] = 0.
        results[model_name][auc_df_std] = auc_array.std()

        results[model_name][auc_df_all] = auc_array


    if show_result:
        display(pd.DataFrame(results))

    return results


def ttest(result_dict, key='auc_df_all'):
    import scipy.stats
    ttest_p_value = {}
    for i, key_i in enumerate(result_dict.keys()):
        ttest_p_value[key_i] = {}
        for j, key_j in enumerate(result_dict.keys()):
            _, pvalue = scipy.stats.ttest_ind(
                result_dict[key_i][key],
                result_dict[key_j][key])
            ttest_p_value[key_i][key_j] = pvalue

    display(pd.DataFrame(ttest_p_value))
    print 'T-Test PValue'



def class_spread(class_means, zs, ys, verbose=False):
    assert class_means.shape[0] == ys.shape[1]
    class_spread = np.zeros(ys.shape[1])
    for i in range(ys.shape[1]):
        dist_mean = np.square(zs[ys[:, i].astype(bool)] - class_means[i]).sum(axis=1)
        class_spread[i] = dist_mean.mean()

    if verbose:
        np.set_printoptions(precision=5)
        print 'Per class avg spread', class_spread
        print 'Overall avg spread', class_spread.mean()
        print 'Overall avg spread', np.max(class_spread)
    else:
        return class_spread.mean(), np.max(class_spread)

def class_separation(class_means, verbose=False):
    n_class = class_means.shape[0]
    all_pair_inter_dist = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            if i == j:
                continue
            all_pair_inter_dist[i, j] = (np.square(class_means[i] - class_means[j])).sum()

    if verbose:
        np.set_printoptions(precision=5)
        print all_pair_inter_dist
        print 'average_inter_dist=' ,  all_pair_inter_dist.mean()
        print 'min_inter_dist=' ,  np.amin(all_pair_inter_dist[all_pair_inter_dist > 0.])
    else:
        return all_pair_inter_dist.mean(), np.amin(all_pair_inter_dist[all_pair_inter_dist > 0.])
