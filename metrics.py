import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def open_classification_performance(score=None, threshold=None, pred_y=None, true_y=None):
    if score is not None and threshold is not None:
        pred_y[score > threshold] = np.amax(pred_y) + 1


    print '', metrics.confusion_matrix(true_y, pred_y)
    print 'Pred Acc:', metrics.accuracy_score(true_y, pred_y)
    p, r, f, _ = metrics.precision_recall_fscore_support(true_y, pred_y)

    print 'Known:\nPre={0:4.2}\tRec={1:4.2}\tF-score={2:4.2}'.format(p[:-1].mean(),
                                                                     r[:-1].mean(),
                                                                     f[:-1].mean())

    print 'Unknown:\nPre={0:4.2}\tRec={1:4.2}\tF-score={2:4.2}'.format(p[-1].mean(),
                                                                       r[-1].mean(),
                                                                       f[-1].mean())


def open_set_classification_metric(true_y, pred_y, is_openset=True, acc=None):
    """It is assumed here that the last class label indicates unknown_class."""
    print '', metrics.confusion_matrix(true_y, pred_y)
    acc = metrics.accuracy_score(true_y, pred_y)
    print 'Pred Acc:', acc
  #  p, r, f, _ = metrics.precision_recall_fscore_support(true_y, pred_y)
    p, r, f, _ = metrics.precision_recall_fscore_support(true_y, pred_y, average="weighted")


    if is_openset:
        print 'Known:\nPre={0:4.2}\tRec={1:4.2}\tF-score={2:4.2}'.format(p[:-1].mean(), r[:-1].mean(), f[:-1].mean())

        print 'Unknown:\nPre={0:4.2}\tRec={1:4.2}\tF-score={2:4.2}'.format(p[-1].mean(), r[-1].mean(), f[-1].mean())

    print 'All:\nPre={0:4.2}\tRec={1:4.2}\tF-score={2:4.2}'.format(p.mean(), r.mean(), f.mean())
    return acc



def auc(y_true, y_score, pos_label=1, plot=False, loc = 'lower right', max_fpr=1.0, figsize=[6,4],
        plot_threshold=False):
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    #%matplotlib inline
    import matplotlib.pyplot as plt
    # To avoid type 3 fonts. ACM Digital library complain about this
    # based on the recomendations here http://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
#    area = metrics.roc_auc_score(y_true, y_score, average="weighted")
    area = metrics.auc(fpr[fpr<=max_fpr], tpr[fpr<=max_fpr])

    if plot:
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label='(auc: {0:.{precision}})'.format(area,
                                                                  precision=2))
        plt.grid(True)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim([0., max_fpr])
        plt.legend(loc=loc)

        if plot_threshold:
            ax2 = plt.twiny()

            # Move twinned axis ticks and label from top to bottom
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")

            # Offset the twin axis below the host
            ax2.spines["bottom"].set_position(("axes", -0.15))

            # Turn on the frame for the twin axis, but then hide all
            # but the bottom spine
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)
            for sp in ax2.spines.itervalues():
                sp.set_visible(False)
            ax2.spines["bottom"].set_visible(True)


            partitions = 11

            from bisect import bisect_left
            def find_lt(a, x):
                'Find rightmost value less than x'
                i = bisect_left(a, x)
                if i:
                    return i

                return 0

            t = []
            for f in np.linspace(0, max_fpr, partitions):
                idx = min(len(fpr)-1, find_lt(fpr[fpr<=max_fpr],f))
                t.append('{0:5.3}'.format(float(threshold[idx])))

            ax2.set_xticks(np.arange(partitions))
#             ax2.set_xticklabels(['{0:5.3}'.format(float(t)) for i, t in enumerate(threshold) \
#                                  if i % (len(threshold) / partitions) == 0])
            ax2.set_xticklabels(t)
            ax2.set_xlabel("Threshold")

        plt.show()
    return area



def gen_color_map(keys):
    colors = cm.rainbow(np.linspace(0, 1, len(keys)))
    return dict(zip(keys, colors))


def eval_clustering(model, xs, ys, n_class, n_cluster, table=False, loc='upper left'):
    """
    Args:
    model - clustering model obejct with predict method
    xs - data to be clustered
    ys - true class labels
    n_class - number of true classes
    n_cluster - number of clusters
    table - If true, then display cluster size data in table format as well
    loc - figure legend location.
    """
    y_pred = model.predict(xs)

    # homogeneity: each cluster contains only members of a single class.
    homogeneity = metrics.homogeneity_score(ys, y_pred)
    print 'homogeneity = ', homogeneity
    # completeness: all members of a given class are assigned to the same cluster.
    completeness = metrics.completeness_score(ys, y_pred)
    print 'completeness = ', completeness

    unique_pred, count_pred = np.unique(y_pred, return_counts=True)
    unique_ys = np.unique(ys)

    cm = np.zeros((n_class, n_cluster), dtype=int)
    class_idx = dict(zip(range(n_class), range(n_class))) #{unique_ys[i]:i for i in xrange(len(unique_ys))}
    cluster_idx = dict(zip(range(n_cluster), range(n_cluster))) # {unique_pred[i]:i for i in xrange(len(unique_pred))}

    for cid in cluster_idx.iterkeys():
        c_memebers = ys[y_pred == cid]
        unique_labels, count_lables = np.unique(c_memebers, return_counts=True)
        for l, c in zip(unique_labels, count_lables):
            cm[class_idx[l], cid] = c

    if table:
        import pandas as pd
        from IPython.display import display
        display(pd.DataFrame(np.concatenate([unique_pred.reshape((len(unique_pred), 1)),
                                             count_pred.reshape((len(count_pred), 1))], axis=1),
        #                      index=rows,
                             columns=['Cluster IDs', 'ClusterSize']))

    color_map = gen_color_map(list(class_idx.iterkeys()))

    bottom = np.zeros(cm.shape[1])
    for c, idx in class_idx.iteritems():
        plt.bar(list(cluster_idx.iterkeys()), cm[idx, :], color=color_map[c], label=str(c),
                bottom=bottom)
        bottom += cm[idx, :]

    plt.xlabel('Cluster IDs')
    plt.ylabel('Cluster Size')
    plt.xticks(list(cluster_idx.iterkeys()), [str(k) for k in cluster_idx.iterkeys()], rotation=90, ha='left')
    plt.xlim((np.amin(unique_pred)-1, np.amax(unique_pred)+1))
    plt.ylim((0, np.amax(count_pred)))
    plt.tight_layout()
    plt.legend(loc=loc)
    plt.show()
    return homogeneity, completeness
