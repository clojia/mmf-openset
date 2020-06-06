import os.path
import pickle
import sys
import time

import numpy as np
import scipy
import scipy.misc
from sklearn import metrics

class OpenKMeans(object):
    def __init__(self, y_dim, transformer=None, contamination=0.01, random_state=None):
        """
        Args:
        @param y_dim: number of clusters.
        """
        self.y_dim = y_dim
        self.transformer = transformer
        self.random_state = random_state
        self.contamination = contamination

        self.model_params = ['y_dim', 'random_state', 'contamination', 'transformer']

    def model_config(self):
        config = {}
        for field, val in vars(self).items():
            if field in self.model_params and field != 'transformer':
                config[field] = val
            elif field in self.model_params and field == 'transformer':
                if  hasattr(val, 'model_config'):
                    config[field] = val.model_config()
                else:
                    config[field] = str(val)
        return config

    def fit(self, X, y=None):
        if self.transformer is not None:
            print 'Fit Transformer'
            self.transformer.fit(X, y)

        from sklearn import cluster
        print 'Fitting KMeans'
        self.kmeans = cluster.KMeans(n_clusters=self.y_dim, random_state=self.random_state, n_jobs=-1)
        self.kmeans.fit(X if self.transformer is None else self.transformer.latent(X))
        self._estimate_threshold(X)

    def _estimate_threshold(self, X):
        dist_from_c = self.decision_function(X)
        cutoff_idx = max(1, int(dist_from_c.shape[0] * self.contamination))
        self.threshold = sorted(dist_from_c)[-cutoff_idx]

    def distance_from_all_classes(self, X):
        return self.kmeans.transform(X if self.transformer is None else self.transformer.latent(X))

    def decision_function(self, X):
        return np.amin(self.distance_from_all_classes(X), axis=1)

    def predict(self, X):
        return self.kmeans.predict(X if self.transformer is None else self.transformer.latent(X))

    def predict_open(self, X):
        dist = self.decision_function(X)
        pred = self.kmeans.predict(X if self.transformer is None else self.transformer.latent(X))
        pred[dist > self.threshold] = self.y_dim
        return pred

class ISF(object):
    def __init__(self, contamination=0.01, n_estimators=100, transformer=None, random_state=None):
        """
        Args:
        @param y_dim: number of clusters.
        """
        self.random_state = random_state
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.transformer = transformer

        self.model_params = ['random_state', 'contamination', 'n_estimators', 'transformer']


    def model_config(self):
        config = {}
        for field, val in vars(self).items():
            if field in self.model_params and field != 'transformer':
                config[field] = val
            elif field in self.model_params and field == 'transformer':
                if  hasattr(val, 'model_config'):
                    config[field] = val.model_config()
                else:
                    config[field] = str(val)
        return config

    def fit(self, X, y=None):
        if self.transformer is not None:
            print 'Fit Transformer'
            self.transformer.fit(X, y)
        from sklearn import ensemble
        print 'Fitting ISF'
        self.isf = ensemble.IsolationForest(n_estimators=self.n_estimators, max_samples='auto',
                                            contamination=self.contamination,
                                            n_jobs=-1, random_state=self.random_state)
        self.isf.fit(X if self.transformer is None else self.transformer.latent(X))
        self._estimate_threshold(X)

    def _estimate_threshold(self, X):
        dist_from_c = self.decision_function(X)
        cutoff_idx = max(1, int(dist_from_c.shape[0] * self.contamination))
        self.threshold = sorted(dist_from_c)[-cutoff_idx]

    def distance_from_all_classes(self, X):
        return None

    def decision_function(self, X):
        return -self.isf.decision_function(X if self.transformer is None else self.transformer.latent(X))

    def predict(self, X):
        """
        Return: 1 for outlier or 0 for inlier.
        """
        pred = self.isf.predict(X if self.transformer is None else self.transformer.latent(X))
        pred[pred == 1] = 0 # inlier
        pred[pred == -1] = 1 # outlier
        return pred

    def predict_open(self, X):
        return self.predict(X)
