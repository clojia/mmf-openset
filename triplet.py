# triplet loss
import tensorflow.keras.backend as K
from itertools import permutations
import random
import tensorflow as tf

import numpy as np

def generate_triplet(x, y,  ap_pairs=10, an_pairs=10):
    data_xy = tuple([x, y])

    trainsize = 1

    triplet_train_pairs = []
    y_triplet_pairs = []
    #triplet_test_pairs = []
    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx, 2)), k=ap_pairs)  # Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx), k=an_pairs)

        # train
        A_P_len = len(A_P_pairs)
        #Neg_len = len(Neg_idx)
        for ap in A_P_pairs[:int(A_P_len * trainsize)]:
            Anchor = data_xy[0][ap[0]]
            y_Anchor = data_xy[1][ap[0]]
            Positive = data_xy[0][ap[1]]
            y_Pos = data_xy[1][ap[1]]
            for n in Neg_idx:
                Negative = data_xy[0][n]
                y_Neg = data_xy[1][n]
                triplet_train_pairs.append([Anchor, Positive, Negative])
                y_triplet_pairs.append([y_Anchor, y_Pos, y_Neg])
                # test

    return np.array(triplet_train_pairs), np.array(y_triplet_pairs)


def triplet_loss(y_true, y_pred, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    anchor = y_pred[:, 0:int(total_lenght * 1 / 3)]
    positive = y_pred[:, int(total_lenght * 1 / 3):int(total_lenght * 2 / 3)]
    negative = y_pred[:, int(total_lenght * 2 / 3):int(total_lenght * 3 / 3)]

    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor - positive), axis=1)

    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor - negative), axis=1)

    # compute loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)

    return loss


def triplet_center_loss(y_true, y_pred, n_classes=10, alpha=0.4):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)
    print('y_true.shap= ', y_true)
    total_lenght = y_pred.shape.as_list()[-1]
    #nprint('total_lenght=',  total_lenght)
    #     total_lenght =12

    # repeat y_true for n_classes and == np.arange(n_classes)
    # repeat also y_pred and apply mask
    # obtain min for each column min vector for each class

    classes = tf.range(0, n_classes,dtype=tf.float32)
    print('classes....')
    print(classes)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)
    print('y_pred_r....')
    print(y_pred_r)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)

    mask = tf.equal(y_true_r[:, :, 0], classes)
    print('mask.....')
    print(mask)
    #mask2 = tf.ones((tf.shape(y_true_r)[0], tf.shape(y_true_r)[1]))  # todo inf

    # use tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32) #+ (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
    masked = tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    minimums = tf.math.reduce_min(masked, axis=1)
    print("minimums....")
    print(minimums)

    loss = K.max(y_pred - minimums +alpha ,0)
    print('tc_loss....')
    print(loss)

    # obtain a mask for each pred


    return loss


def center_loss(embedding, labels, num_classes, name=''):
    '''
    embedding dim : (batch_size, num_features)
    '''
    batch_size=100

    with tf.variable_scope(name):
        num_features = embedding.get_shape()[1]
        centroids = tf.get_variable('c', shape=[num_classes, num_features],
                                    dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(),
                                    trainable=False)
        centroids_delta = tf.get_variable('centroidsUpdateTempVariable', shape=[num_classes, num_features],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer(), trainable=False)

        centroids_batch = tf.gather(centroids, labels)

        cLoss = tf.nn.l2_loss(embedding - centroids_batch) / float(batch_size)  # Eq. 2

        diff = centroids_batch - embedding

        delta_c_nominator = tf.scatter_add(centroids_delta, labels, diff)

        indices = tf.expand_dims(labels, -1)
        updates = tf.constant(value=1, shape=[indices.get_shape()[0]], dtype=tf.float32)
        shape = tf.constant([num_classes])
        labels_sum = tf.expand_dims(tf.scatter_nd(indices, updates, shape), -1)

        centroids = centroids.assign_sub(ALPHA * delta_c_nominator / (1.0 + labels_sum))

        centroids_delta = centroids_delta.assign(tf.zeros([num_classes, num_features]))

        '''
        # The same as with scatter_nd
        one_hot_labels = tf.one_hot(y,num_classes)
        labels_sum = tf.expand_dims(tf.reduce_sum(one_hot_labels,reduction_indices=[0]),-1)
        centroids = centroids.assign_sub(ALPHA * delta_c_nominator / (1.0 + labels_sum))
        '''
        return cLoss, centroids

