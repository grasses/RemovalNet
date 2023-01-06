#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/10/10, homeway'


import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def rename_labels(labels):
    for idx, name in enumerate(labels):
        if "1" == name:
            labels[idx] = "L1"
        if "2" == name:
            labels[idx] = "L2"
        if "negative" in name:
            labels[idx] = "Negative"
        if "prune" in name:
            ratio = name.split("(")[-1].split(")")[0]
            labels[idx] = f"WP({ratio})"
        if "distill" in name:
            labels[idx] = "Distill"
        if "steal" in name:
            labels[idx] = "Steal"
        if "quantize" in name:
            qtype = name.split("(")[-1].split(")")[0]
            labels[idx] = f"WQ({qtype})"
        if "finetune" in name:
            ratio = name.split(",")[-1].split(")")[0]
            labels[idx] = f"FT({ratio})"
        if "removal" in name:
            labels[idx] = "RemovalNet"
        if "DDM" in name:
            labels[idx] = labels[idx].replace("DDM", "DDV")
    return labels


def sigmoid(z):
    """ this function implements the sigmoid function, and
    expects a numpy array as argument """
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return sigmoid


def normalize_dict(x, reverse=False):
    x = np.array(x)
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x-x_min) / (x_max-x_min)
    x = sigmoid(x)
    if reverse:
        x = 1-x
    return x.tolist()


def plot_boxplot(data, xticklabels, fpath=None, fontsize=20):
    figure, ax = plt.subplots(figsize=(24, 10), dpi=200)
    ax.yaxis.grid(True)

    off = -0.2
    legends, bplots = [], []
    for idx, (lenged, item) in enumerate(data.items()):
        bplot = ax.boxplot(item, widths=0.1, showfliers=False, patch_artist=True, positions=np.arange(len(item))+off)
        legends.append(lenged)
        bplots.append(bplot["boxes"][0])
        for patch in bplot["boxes"]:
            patch.set_facecolor(colors[idx])
        off += 0.4

    legends = rename_labels(legends)
    ax.legend(bplots, legends, loc='best', fontsize=fontsize-10)
    plt.setp(ax, xticks=np.arange(len(item)), xticklabels=rename_labels(xticklabels))
    plt.xticks(fontsize=fontsize-10)
    plt.yticks(fontsize=fontsize-10)
    print(f"-> save figugre:{fpath}")
    plt.savefig(fpath)



def boxplot_distance(dists, metrics, xticks, ylabel, fpath, fontsize=32):
    off_x = 0
    bplots, legends = [], []
    figure, ax = plt.subplots(figsize=(20, 10), dpi=160)
    ax.yaxis.grid(True)


    gap = 0.4
    m = list(metrics)[0]
    if "IPGuard" in m or "ModelDiff" in m:
        gap = 0.2

    for idx, metric in enumerate(metrics):
        item = dists[idx].tolist()
        bplot = ax.boxplot(item, widths=0.15, showfliers=False, patch_artist=True,
                           boxprops=dict(linestyle='-', linewidth=1.3, color='black'),
                           medianprops=dict(linestyle='-.', linewidth=1.5, color='black'),
                           positions=np.arange(len(item)) - gap + off_x)
        legends.append(metric)
        bplots.append(bplot["boxes"][0])

        off_x += 0.2
        for patch in bplot["boxes"]:
            patch.set_facecolor(colors[idx])
    legends = rename_labels(legends)
    ax.legend(bplots, legends, loc='best', fontsize=fontsize)
    plt.setp(ax, xticks=np.arange(len(item)), xticklabels=rename_labels(xticks))
    plt.xticks(fontsize=fontsize - 8)
    plt.yticks(fontsize=fontsize - 8)
    plt.ylabel(ylabel, fontsize=fontsize - 8)
    print(f"-> save figugre:{fpath}")
    plt.savefig(fpath)

