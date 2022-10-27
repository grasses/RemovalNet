#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import copy
import torch
import numpy as np
import os.path as osp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import helper
from torchcam.methods import LayerCAM
sys_args = helper.get_args()


def plot_embedding(xy, file_path, lims=100, fontsize=30):
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    plt.grid()
    size = int(len(xy) / 2)
    labels = ["model_0", "model_t"]
    for idx in range(2):
        off = int(idx * size)
        plt.scatter(xy[off:off + size, 0], xy[off:off + size, 1], lw=4, s=60, label=labels[idx])

    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="upper right", numpoints=1, fontsize=fontsize)
    plt.savefig(file_path)
    print(f"-> saving fig: {file_path}")


def pixel_plot(data, ax, fontsize=18, hide_labels=False):
    pc = ax.pcolormesh(data, vmin=0, vmax=1)
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    return pc


def view_learning_state(data, file_path, fontsize=30):
    for key in data["keys"]:
        plt.figure(figsize=(16, 12), dpi=100)
        plt.cla()
        plt.grid()
        x = np.array(data["t"], dtype=np.int32)
        y = np.array(data[key])
        plt.plot(x, y, label=f"model_t", linewidth=6, marker="*", markersize=10, linestyle="solid")
        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel(key.upper(), fontsize=fontsize)
        if key == "acc":
            plt.ylim(0, 100.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="lower left", numpoints=1, fontsize=fontsize)
        fpath = file_path + f"_{key.upper()}.pdf"
        plt.savefig(fpath)
        print(f"-> saving fig: {fpath}")


def view_layer_activation(model_0, model_t, x, y, target_layer, fig_path, size=8, fontsize=30, device=torch.device("cuda:1")):
    x = x.clone()
    model_0 = copy.deepcopy(model_0).to(device)
    model_t = copy.deepcopy(model_t).to(device)

    class_idx = y.tolist()
    CAM0 = LayerCAM(model_0, target_layer)
    cams = CAM0(class_idx=class_idx, scores=model_0(x))
    fused_cam0 = CAM0.fuse_cams(cams).detach().cpu()
    CAM0.remove_hooks()

    CAM1 = LayerCAM(model_t, target_layer)
    cams = CAM1(class_idx=class_idx, scores=model_t(x))
    fused_camt = CAM1.fuse_cams(cams).detach().cpu()
    CAM1.remove_hooks()

    fig = plt.figure(constrained_layout=True, figsize=(12, 2 + size * 4), dpi=160)
    subfigs = fig.subfigures(1, 2, wspace=0.07, hspace=0.07)
    axsLeft = subfigs[0].subplots(size, 1, sharey=True)
    subfigs[0].set_facecolor('#ffffff')
    for nn, ax in enumerate(axsLeft):
        fmap = fused_cam0[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[0].suptitle(f'model_0 {target_layer}', fontsize=fontsize-5)
    subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

    axsRight = subfigs[1].subplots(size, 1, sharex=True)
    for nn, ax in enumerate(axsRight):
        fmap = fused_camt[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[1].set_facecolor('#ffffff')
    subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight, location='bottom')
    subfigs[1].suptitle(f'model_t {target_layer}', fontsize=fontsize-5)
    fig.suptitle(f'LayerCAM of {target_layer}', fontsize=fontsize)
    fig_path = fig_path + f".pdf"
    print(f"-> save LayerCam:{fig_path}")
    plt.savefig(fig_path)


def view_decision_boundary(model1, model2, test_loader, fig_path, step, device=sys_args.device):
    model1 = model1.to(device)
    model2 = model2.to(device)

    logist1, logist2 = [], []
    select_t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    select_logist1, select_logist2 = [], []
    correct1, correct2, cnt = 0, 0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out1 = model1(x).detach().cpu()
            out2 = model2(x).detach().cpu()
            if i == 0:
                if out1.shape[1] < 100:
                    select_t = [1, 2, 3, 4]

            for t in select_t:
                idxs = (y == t).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    select_logist1.append(out1[idxs])
                    select_logist2.append(out2[idxs])
                if i > 15:
                    break

            if i <= 5:
                logist1.append(out1)
                logist2.append(out2)
                correct1 += y.cpu().eq(out1.argmax(dim=1).view_as(y)).sum()
                correct2 += y.cpu().eq(out2.argmax(dim=1).view_as(y)).sum()
                cnt += len(x)

    acc1 = round(100.0 * float(correct1 / cnt), 3)
    acc2 = round(100.0 * float(correct2 / cnt), 3)

    logist_array = torch.cat(logist1 + logist2).numpy()
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(logist_array)
    plot_embedding(xy, file_path=f"{fig_path}_t{step}.pdf", lims=150)

    select_logist_array = torch.cat(select_logist1 + select_logist2).detach().cpu().numpy()
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(select_logist_array)
    name = ""
    for k in select_t:
        name += f"{k}-"
    file_path = osp.join(fig_path, f"{fig_path}_C{name[:-1]}_t{step}.pdf")
    plot_embedding(xy, lims=90, file_path=file_path)
    return acc1, acc2











