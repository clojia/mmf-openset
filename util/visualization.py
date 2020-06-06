import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.manifold import TSNE
import seaborn as sns
from scipy import stats

def gen_color_map(keys):
    colors = cm.nipy_spectral(np.linspace(0, 1, len(keys)))
    return dict(zip(keys, colors))

def visualize_dataset_2d(x1, x2, ys, alpha=0.5, x1_label='', x2_label='',
                         loc='upper left', figsize=(16, 8), xlim=None, ylim=None,
                         unique_ys=None, save_path=None, label_text_lookup=None):
    """
    Args:
    x1 - data's first dimention
    x2 - data's second dimention

    """

    # To avoid type 3 fonts. ACM Digital library complain about this
    # based on the recomendations here http://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if unique_ys is not None:
        c_lookup = gen_color_map(unique_ys)
    else:
        c_lookup = gen_color_map(set(ys))

    #     c_sequence = [''] * len(ys)
    #     for i in xrange(len(ys)):
    #         c_sequence[i] = c_lookup[ys[i]]

    plt.figure(figsize=figsize)
    for label in set(ys):
        color = c_lookup[label]
        mask = ys == label
        plt.scatter(x1[mask], x2[mask], c=color,
                    label=label if label_text_lookup is None else label_text_lookup[label],
                    alpha=alpha)

    #plt.scatter(x1, x2, c=c_sequence, alpha=alpha)
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    lgd=plt.legend(loc=loc)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()

def qqplot(x, y, **kwargs):
    _, xr = stats.scatterplot(x, fit=False)
    _, yr = stats.scatterplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)

def visualize_t_SNE(X, ys, ts_known_mask=None, grid_shape=(2,2), alpha=0.5, xlim=None, ylim=None,
                    loc='upper left', bbox_to_anchor=(1.04,1), figsize=(16, 8),
                    unique_ys=None, save_path=None, label_text_lookup=None):

    labels = []
    for label in ys:
        labels.append(label_text_lookup[label])

    c_lookup = gen_color_map(label_text_lookup.values())
   # palette = sns.color_palette("bright", len(set(labels)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))



    tsne_df = pd.DataFrame({'t0': tsne_results[:, 0],
                            't1': tsne_results[:, 1],
                            'digit': labels,
                            'known': ts_known_mask})

    plt.figure()
    s = sns.FacetGrid(tsne_df, hue="digit", col="known", height=4)
    s.map(plt.scatter, "t0", "t1", alpha=.5, linewidth=.3, edgecolor="white")
    s.set(xlim=(-90, 90), ylim=(-90, 90))
    s.add_legend()
    if save_path:
        plt.savefig(save_path)

  #  s = sns.scatterplot(x="t0", y="t1", hue="digit", alpha=0.5,
      #        palette = c_lookup,
    #          legend='full',
      #        data=tsne_df)

   # s.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=1)
    #s.set(xlim=xlim,ylim=ylim)



# colors = c_lookup
 #   target_ids = range(len(ys))
   # for i, c, label in zip(target_ids, colors, set(ys)):
     #   plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=c, label=label if label_text_lookup is None else label_text_lookup[label])


  #  plt.legend()
  #  plt.show()

def visualize_dataset_nd(X, ys, grid_shape=(2,2), alpha=0.5, xlim=None, ylim=None,
                         loc='upper left', bbox_to_anchor=(1.04,1), figsize=(16, 8),
                         unique_ys=None, save_path=None, label_text_lookup=None):
    """
    Args:
    X: 2d np.array
    ys: 1d n.array

    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # To avoid type 3 fonts. ACM Digital library complain about this
    # based on the recomendations here http://phyletica.org/matplotlib-fonts/
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if unique_ys is not None:
        c_lookup = gen_color_map(unique_ys)
    else:
        c_lookup = gen_color_map(set(ys))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(grid_shape[0], grid_shape[1])

    n_dim = X.shape[1]
    dim_1 = 0
    dim_2 = 1

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            ax = fig.add_subplot(gs[i, j])
            for label in set(ys):
                color = c_lookup[label]
                mask = ys == label
                ax.scatter(X[mask, dim_1], X[mask, dim_2], c=color,
                           label=label if label_text_lookup is None else label_text_lookup[label],
                           alpha=alpha)
                ax.set_xlabel('Z{0}'.format(dim_1))
                ax.set_ylabel('Z{0}'.format(dim_2))
                ax.grid(True)
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)


            dim_2 += 1
            if dim_2 == n_dim:
                dim_1 += 1
                dim_2 = dim_1 + 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_z_separate(z, ts_y, ts_known_mask,
                      n_scatter=1000, unique_ys=range(7), xlim=None, ylim=None,
                      grid_shape=(1,3), figsize=(12, 4),font_size=13, markersize=None,
                      save_path_known=None,
                      save_path_unknown=None,
                      label_text_lookup=None):
    import matplotlib as mpl
    font = {'family' : 'normal',
    #         'weight' : 'bold',
            'size'   : font_size}

    mpl.rc('font', **font)

    def plot(z, ys, path):
        if z.shape[1] == 2:
            visualize_dataset_2d(z[:, 0], z[:, 1], ys,  xlim=xlim, ylim=ylim,
                                 alpha=0.5, figsize=(8, 6), unique_ys=unique_ys, save_path=path,
                                 label_text_lookup=label_text_lookup)
        elif z.shape[1] == 3:
            visualize_dataset_nd(z, ys, grid_shape=(1,3), alpha=0.5, xlim=xlim, ylim=ylim,
                                 loc='upper left', bbox_to_anchor=(1.04,1), figsize=(12, 4),
                                 unique_ys=unique_ys, save_path=path,
                                 label_text_lookup=label_text_lookup)
      #      visualize_t_SNE(z, ys, grid_shape=(1, 3), alpha=0.5, xlim=xlim, ylim=ylim,
         #                   loc='upper left', bbox_to_anchor=(1.04, 1), figsize=(12, 4),
        #                    unique_ys=unique_ys, save_path=path, label_text_lookup=None)


        else:
       #     visualize_dataset_nd(z, ys, grid_shape=grid_shape, alpha=0.5, xlim=xlim, ylim=ylim,
            #                     loc='upper left', bbox_to_anchor=(1.04,1), figsize=figsize,
           #                      unique_ys=unique_ys, save_path=path,
            #                     label_text_lookup=label_text_lookup)
            visualize_t_SNE(z, ys, grid_shape=grid_shape, alpha=0.5, xlim=xlim, ylim=ylim,
                            loc='upper left', bbox_to_anchor=(1.04, 1), figsize=figsize,
                            unique_ys=unique_ys, save_path=path, label_text_lookup=label_text_lookup)

    z = z[:n_scatter]
    y = np.argmax(ts_y[:n_scatter], axis=1)
    known_mask = ts_known_mask[:n_scatter]
    unknown_mask = np.logical_not(known_mask)
    #plot known
    plot(z[known_mask], y[known_mask], save_path_known)


    #plot unknown
    plot(z[unknown_mask], y[unknown_mask], save_path_unknown)



    mpl.rcdefaults()
