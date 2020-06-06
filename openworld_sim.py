from sets import Set
import random
import numpy as np
from sklearn import preprocessing

class OpenWorldSim(object):
    """ Create an openworld scenario using the given dataset.
    """
    def __init__(self, train_data, train_lable,
                 val_data=None, val_label=None,
                 test_data=None, test_label=None,
                 tr_classes=6, seed=None):
        np.random.seed(seed)
        if len(train_lable.shape) == 1:
            classes = np.unique(train_label)
        else:
            classes = np.arange(train_lable.shape[1])

        if isinstance(tr_classes, list):
            self.tr_classes =  tr_classes
            self.unknown_classes = [c for c in classes if c not in self.tr_classes]
        else:
            np.random.shuffle(classes)
            self.unknown_classes = classes[tr_classes:]
            self.tr_classes =  classes[:tr_classes]


        self.tr_mask = self.mask_unknown(train_lable, self.tr_classes)
        self.val_mask = self.mask_unknown(val_label, self.tr_classes)
        self.ts_mask = self.mask_unknown(test_label, self.tr_classes)

        self.tr_data = train_data
        self.tr_lable = train_lable
        self.val_data = val_data
        self.val_label = val_label
        self.ts_data = test_data
        self.ts_label = test_label

    def mask_unknown(self, label, tr_classes):
        if label is None:
            return None

        mask = np.array([False] * len(label))
        for c in tr_classes:
            if len(label.shape) == 1:
                mask = np.logical_or(mask, label==c)
            else:
                mask = np.logical_or(mask, label[:,c]==1)

        return mask

    def _del_unknown_class(self, label):
        return np.delete(label, self.unknown_classes, axis=1)

    def train_label(self):
        if len(self.tr_lable.shape) == 1:
            label = self.tr_lable[self.tr_mask]
        else:
            label = self.tr_lable[self.tr_mask]
        print("train_label len: " + str(train_label))
        
        return self._del_unknown_class(label)

    def train_data(self):
        print("train_data len: " + str(self.tr_data[self.tr_mask]))
        return self.tr_data[self.tr_mask]

    def _val_test(self, label, mask):
        if len(label.shape) == 1:
            label[np.logical_not(mask)] = -1
        else:
            label[np.logical_not(mask)] = 0
            unknown = np.zeros(label.shape[0])[:, None]
            unknown[np.logical_not(mask)] = 1
            label = np.concatenate((label, unknown), axis=1)

        return self._del_unknown_class(label)

    def validation_label(self):
        return self._val_test(np.copy(self.val_label), self.val_mask)

    def validation_data(self):
        return self.val_data

    def test_label(self):
        return self._val_test(np.copy(self.ts_label), self.ts_mask)

    def test_data(self):
        return self.ts_data


class OpenWorldMsData(object):
    """ Create an openworld scenario using the given dataset.
    """
    def __init__(self, data, label,
                 tr_classes=6, comb_val_test=False, seed=None,
                 normalize=False):
        """
            :param label: assumes class label starting from 0 for non-onehot ecoded
        """
        self.comb_val_test = comb_val_test
        np.random.seed(seed)
        assert len(label.shape) == 1
        classes = np.unique(label)

        if isinstance(tr_classes, list):
            self.tr_classes =  tr_classes
            self.unknown_classes = [c for c in classes if c not in self.tr_classes]
        else:
            np.random.shuffle(classes)
            self.unknown_classes = classes[tr_classes:].astype(int)
            self.tr_classes =  classes[:tr_classes].astype(int)

        enc = preprocessing.OneHotEncoder(n_values=len(self.tr_classes) + len(self.unknown_classes), sparse=False)
        enc.fit(label.reshape(-1, 1))

        tr_x, tr_y, _, _, ts_x, ts_y, tr_classes, val_classes, ts_classes = \
            separate_dataset(data, label, list(self.tr_classes), [], list(self.unknown_classes),
                             tvt_ratio=[0.75, 0.0, 0.25], vt_ratio=[0.0, 1.0],  seed=seed)

        split_point = int(ts_x.shape[0] * 0.33)
        idx = np.arange(ts_x.shape[0])
        np.random.shuffle(idx)
        ts_x = ts_x[idx]
        ts_y = ts_y[idx]

        self.tr_data = preprocessing.normalize(tr_x) if normalize else tr_x
        self.tr_label = self._del_unknown_class(enc.transform(tr_y.reshape(-1, 1)))

        self.val_data = preprocessing.normalize(ts_x[:split_point]) if normalize else ts_x[:split_point]
        self.val_label = enc.transform(ts_y[:split_point].reshape(-1, 1))
        val_mask = self.mask_unknown(self.val_label, self.tr_classes)
        self.val_label = self._val_test(self.val_label, val_mask)

        self.ts_data = preprocessing.normalize(ts_x[split_point:]) if normalize else ts_x[split_point:]
        self.ts_label = enc.transform(ts_y[split_point:].reshape(-1, 1))
        ts_mask = self.mask_unknown(self.ts_label, self.tr_classes)
        self.ts_label = self._val_test(self.ts_label, ts_mask)


    def mask_unknown(self, label, tr_classes):
        if label is None:
            return None

        mask = np.array([False] * len(label))
        for c in tr_classes:
            c = int(c)
            if len(label.shape) == 1:
                mask = np.logical_or(mask, label==c)
            else:
                mask = np.logical_or(mask, label[:,c]==1)

        return mask

    def _del_unknown_class(self, label):
        return np.delete(label, self.unknown_classes, axis=1)

    def _val_test(self, label, mask):
        label[np.logical_not(mask)] = 0
        unknown = np.zeros(label.shape[0])[:, None]
        unknown[np.logical_not(mask)] = 1
        label = np.concatenate((label, unknown), axis=1)

        return self._del_unknown_class(label)

    def train_label(self):
        return self.tr_label

    def train_data(self):
        return self.tr_data

    def validation_label(self):
        return self.val_label

    def validation_data(self):
        return self.val_data

    def test_label(self):
        if self.comb_val_test:
            return np.concatenate((self.val_label, self.ts_label), axis=0)
        else:
            return self.ts_label

    def test_data(self):
        if self.comb_val_test:
            return np.concatenate((self.val_data, self.ts_data), axis=0)
        else:
            return self.ts_data




def class_statistics(ys):
    """ Return class distribution in the given lable list.
    """
    class_stats = {}
    for j in xrange(len(ys)):
        if ys[j] not in class_stats:
            class_stats[int(ys[j])] = 0
        class_stats[int(ys[j])] = class_stats[ys[j]] + 1

    return class_stats

def sample_distributions(all_classes, class_stats, tvt_ratio, vt_ratio, num_tr_classes,
                         num_val_classes):
    """
    Calculate the exact number of samples for each class to be added to train, validation and test

    Args:
    all_classes - all class labeles
    class_stats - dictionary mapping class labele to the total number of samples in that class.
    tvt_ratio - for the classes that are in all three the ratio of samples in
        [training, validation, test]
    vt_ratio - for the classes that are only in validation and test set the
        ratio of samples in [validation, test]
    num_tr_classes - the number of class present in the training where the actual class
        are going to be selected at random.
    num_val_classes - the number of new classes present in the validation where the actual class
        are going to be selected at random.
    """
    num_samples = {}  # num samples in tr,val,ts sets
    num_tr = 0
    num_val = 0
    num_ts = 0
    for i in xrange(len(all_classes)):
        if i < num_tr_classes:
            ns = [0] * 3
            ns[0] = int(tvt_ratio[0] * class_stats[all_classes[i]])
            num_tr += ns[0]
            ns[1] = int(tvt_ratio[1] * class_stats[all_classes[i]])
            num_val += ns[1]
            ns[2] = int(class_stats[all_classes[i]] - ns[0] - ns[1])
            num_ts += ns[2]
            num_samples[all_classes[i]] = ns
        elif i >= num_tr_classes and i < num_tr_classes + num_val_classes:
            ns = [0] * 3
            ns[1] = int(vt_ratio[0] * class_stats[all_classes[i]])
            num_val += ns[1]
            ns[2] = int(class_stats[all_classes[i]] - ns[1])
            num_ts += ns[2]
            num_samples[all_classes[i]] = ns
        else:
            ns = [0] * 3
            ns[2] = class_stats[all_classes[i]]
            num_ts += ns[2]
            num_samples[all_classes[i]] = ns

    return num_samples, int(num_tr), int(num_val), int(num_ts)


def separate_dataset(xs, ys, num_tr_classes, num_val_classes, num_ts_classes,
                     tvt_ratio=[0.5, 0.25, 0.25], vt_ratio=[0.5, 0.5],  seed=None):
    """ Separate dataset it into training, validation, and test sets.
    In this new dataset not all class will be present in training and validation.

    Args:
    xs - feature vectors
    ys - class labels
    all_classes - list of all classes
    dataset - the dataste in the form of an Arrf object
    num_tr_classes - the number of class present in the training where the actual class
        are going to be selected at random.
    num_val_classes - the number of new classes present in the validation where the actual class
        are going to be selected at random.
    num_ts_classes - the number of new class present in the test where the actual class
        are going to be selected at random.
    tvt_ratio - for the classes that are in all three the ratio of samples in
        [training, validation, test] (default [0.5, 0.25, 0.25])
    vt_ratio - for the classes that are only in validation and test set the ratio
        of samples in [validation, test] (default [0.5, 0.5])
    seed=None - seed for RNG

    Returns:
    tr_x - training feature matrix (N_tr, Dimention)
    tr_y - training labels vector (N_tr)
    val_x - validation feature matrix (N_val, Dimention)
    val_y - validation labels vector (N_val)
    ts_x - test feature matrix (N_ts, Dimention)
    tx_y - test labels vector (N_ts)
    tr_classes - class labels in training
    val_classes - class labeles in validation (exclusing training)
    ts_classes - class labeles in test (excluding validation and training)
    """
    random.seed(seed)

    if (isinstance(num_tr_classes, list) and
            isinstance(num_val_classes, list) and
            isinstance(num_ts_classes, list)):
        tr_classes = num_tr_classes
        num_tr_classes = len(tr_classes)
        val_classes = num_val_classes
        num_val_classes = len(val_classes)
        ts_classes = num_ts_classes
        num_ts_classes = len(ts_classes)
        all_classes = tr_classes + val_classes + ts_classes
    elif (isinstance(num_tr_classes, int) and
          isinstance(num_val_classes, int) and
          isinstance(num_ts_classes, int)):
        # all_classes = [dataset.class_info.get_int(c) for c in dataset.class_info.get_noms()]
        all_classes = list(Set(ys))
        all_classes = np.array(all_classes, dtype='int32')
        random.shuffle(all_classes)

        # Pick tr_classes at random and set as tr_classes
        tr_classes = Set([])
        for i in xrange(num_tr_classes):
            tr_classes.add(all_classes[i])

        # Pick num_val_classes of the remaining at random and set as val_classes
        val_classes = Set()
        for i in xrange(num_tr_classes, num_tr_classes + num_val_classes):
            val_classes.add(all_classes[i])

        # Pick the remaiing as test_classes
        ts_classes = Set()
        for i in xrange(num_tr_classes + num_val_classes,
                        num_tr_classes + num_val_classes + num_ts_classes):
            ts_classes.add(all_classes[i])
    else:
        raise Exception('Illegal input for num_tr_classes %s, ' \
                        'num_val_classes %s and num_ts_classes%s.'
                        % (type(num_tr_classes), type(num_val_classes), type(num_ts_classes)))

    # Class stats
    class_stats = class_statistics(ys)

    # Calculate the exact number of samples for each class to be added to
    # train, validation and test
    num_samples, num_tr, num_val, num_ts = sample_distributions(all_classes,
                                                                class_stats,
                                                                tvt_ratio,
                                                                vt_ratio,
                                                                num_tr_classes,
                                                                num_val_classes)

    # Data insatnces at random according to the calculated numbers
    rand_idx = range(len(ys))
    random.shuffle(rand_idx)

    tr_x = np.zeros((num_tr, ) + xs.shape[1:])
    tr_y = np.zeros(num_tr)
    val_x = np.zeros((num_val, ) + xs.shape[1:])
    val_y = np.zeros(num_val)
    ts_x = np.zeros((num_ts, ) + xs.shape[1:])
    ts_y = np.zeros(num_ts)

    i_tr = 0
    i_val = 0
    i_ts = 0
    for i in xrange(len(rand_idx)):
        c = int(ys[rand_idx[i]])
        if num_samples[c][0] > 0:
            tr_x[i_tr] = xs[rand_idx[i]]
            tr_y[i_tr] = c
            i_tr += 1
            num_samples[c][0] = num_samples[c][0] - 1
        elif num_samples[c][1] > 0:
            val_x[i_val] = xs[rand_idx[i]]
            val_y[i_val] = c
            i_val += 1
            num_samples[c][1] = num_samples[c][1] - 1
        elif num_samples[c][2] > 0:
            ts_x[i_ts] = xs[rand_idx[i]]
            ts_y[i_ts] = c
            i_ts += 1
            num_samples[c][2] = num_samples[c][2] - 1

    return tr_x, tr_y, val_x, val_y, ts_x, ts_y, tr_classes, val_classes, ts_classes
