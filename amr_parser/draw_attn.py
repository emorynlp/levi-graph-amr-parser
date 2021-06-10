# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-11-22 17:24
import os
import pickle
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", cbar=True, labels=False, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    if cbar:
        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    if not labels:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def main():
    sent = ['The', 'boy', 'wants', 'the', 'girl', 'to', 'believe', 'him', '.'] + ['BOG', 'want', 'believe', 'ARG1',
                                                                                  'boy',
                                                                                  'ARG0', 'ARG1', 'girl', 'ARG0']
    print(len(sent))
    alignment_weight = torch.load('attn.pt', map_location='cpu')
    alignment_weight = alignment_weight[:, :, 0, :]
    src_len = snt_len = tgt_len = 9
    attn = alignment_weight[1:, :, :].max(dim=0)[0]
    attn[:snt_len, :snt_len] = alignment_weight[:, :snt_len, :snt_len].max(dim=0)[0]
    # attn[-src_len:, :snt_len] = alignment_weight[0, -src_len:, :snt_len]
    # first head
    # attn = alignment_weight[0, :, :]
    attn = attn.cpu().detach().numpy()
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8))
    fig, ax4 = plt.subplots(1, 1, figsize=(8, 8))
    # get rid of BOG
    bog = sent.index('BOG')
    attn[bog, :] = -1
    attn[:, bog] = -1
    attn = attn[attn != -1].reshape((17, 17))
    sent.remove('BOG')
    cmap = "binary"
    tgt_len -= 1
    src_len -= 1

    # im, _ = heatmap(attn[:snt_len, :snt_len], sent[:snt_len], sent[:snt_len], ax=ax4, vmin=0,
    #                 cmap=cmap, cbarlabel="Attention Weight", cbar=False, labels=True)
    # fig.savefig("/Users/hankcs/Dropbox/应用/Overleaf/NAACL-2021-AMR/fig/attn_token_token.pdf",
    #             bbox_inches='tight')

    # im, _ = heatmap(attn[-src_len:, -tgt_len:], sent[-src_len:], sent[-src_len:], ax=ax4, vmin=0,
    #                 cmap=cmap, cbarlabel="Attention Weight", cbar=False, labels=False)
    # fig.savefig("/Users/hankcs/Dropbox/应用/Overleaf/NAACL-2021-AMR/fig/attn_node_node.pdf",
    #             bbox_inches='tight')
    im, _ = heatmap(attn[-src_len:, :snt_len], sent[-src_len:], sent[:snt_len], ax=ax4, vmin=0,
                    cmap=cmap, cbarlabel="Attention Weight", cbar=False, labels=True)
    # fig.savefig("/Users/hankcs/Dropbox/应用/Overleaf/NAACL-2021-AMR/fig/attn_node_token.pdf",
    #             bbox_inches='tight')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
