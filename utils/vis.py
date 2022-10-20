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



def plot_logists(pos, file_path, lims=100, fontsize=30):
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    plt.grid()
    size = int(len(pos) / 2)
    labels = ["model_0", "model_t"]
    for idx in range(2):
        off = int(idx * size)
        plt.scatter(pos[off:off + size, 0], pos[off:off + size, 1], lw=4, s=60, label=labels[idx])

    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="upper right", numpoints=1, fontsize=fontsize)
    plt.savefig(file_path)
    print(f"-> saving fig: {file_path}")


def plot_learning_curve(data, file_path, fontsize=30):
    for key in data["keys"]:
        plt.figure(figsize=(16, 12), dpi=100)
        plt.cla()
        plt.grid()
        x = np.array(data["t"], dtype=np.int32)
        y = np.array(data[key])
        plt.plot(x, y, label=f"model_{data['t'][-1]}", linewidth=6, marker="o", linestyle="solid")
        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel(key.upper(), fontsize=fontsize)
        if key == "acc":
            plt.ylim(0, 100.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="lower left", numpoints=1, fontsize=fontsize)
        fpath = file_path + f"_{key}.pdf"
        plt.savefig(fpath)
        print(f"-> saving fig: {fpath}")



def pixel_plot(data, ax, fontsize=18, hide_labels=False):
    pc = ax.pcolormesh(data, vmin=0, vmax=1)
    if not hide_labels:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    return pc


def plot_layer_cam(model_0, model_t, x, y, target_layer, fig_path, size=10, fontsize=30):
    x = x.clone()
    model_0 = copy.deepcopy(model_0).to(x.device)
    model_t = copy.deepcopy(model_t).to(x.device)

    class_idx = y.clone().cpu().numpy().tolist()
    extractor_0 = LayerCAM(model_0, target_layer)
    cams = extractor_0(class_idx=class_idx, scores=model_0(x))
    fused_cam_0 = extractor_0.fuse_cams(cams).detach().cpu()

    extractor_t = LayerCAM(model_t, target_layer)
    cams = extractor_t(class_idx=class_idx, scores=model_t(x))
    fused_cam_t = extractor_t.fuse_cams(cams).detach().cpu()

    fig = plt.figure(constrained_layout=True, figsize=(12, size * 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    axsLeft = subfigs[0].subplots(size, 1, sharey=True)
    subfigs[0].set_facecolor('0.75')
    for nn, ax in enumerate(axsLeft):
        fmap = fused_cam_0[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[0].suptitle('model_0 featuremap', fontsize=fontsize-5)
    subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

    axsRight = subfigs[1].subplots(size, 1, sharex=True)
    for nn, ax in enumerate(axsRight):
        fmap = fused_cam_t[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[1].set_facecolor('0.85')
    subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight, location='bottom')
    subfigs[1].suptitle('model_t featuremap', fontsize=fontsize-5)
    fig.suptitle(f'LayerCAM of {target_layer}', fontsize=fontsize)

    fig_path = fig_path + f".pdf"
    print(f"-> save LayerCam:{fig_path}")
    plt.savefig(fig_path)


def plot_logist_embedding(teacher, student, test_loader, out_root, file_name, device=sys_args.device):
    teacher = teacher.to(device).train()
    student = student.to(device).train()
    logist1 = []
    logist2 = []

    select_t = [10, 20, 30, 40]
    select_logist1 = []
    select_logist2 = []

    correct1, correct2, cnt = 0, 0, 0
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out1 = teacher(x).detach().cpu()
            out2 = student(x).detach().cpu()

            for t in select_t:
                idxs = (y == t).nonzero(as_tuple=True)[0]
                if len(idxs) > 0:
                    select_logist1.append(out1[idxs])
                    select_logist2.append(out2[idxs])
            logist1.append(out1)
            logist2.append(out2)
            cnt += len(x)
            correct1 += y.cpu().eq(out1.argmax(dim=1).view_as(y)).sum()
            correct2 += y.cpu().eq(out2.argmax(dim=1).view_as(y)).sum()
    acc1 = float(correct1 / cnt)
    acc2 = float(correct2 / cnt)

    logist_array = torch.cat(logist1 + logist2).numpy()
    pos = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(logist_array)

    file_path = osp.join(out_root, file_name)
    plot_logists(pos, file_path=file_path, lims=100)

    select_logist_array = torch.cat(select_logist1 + select_logist2).detach().cpu().numpy()
    pos = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(select_logist_array)

    name = "_"
    for k in select_t:
        name += f"{k}_"
    file_path = osp.join(out_root, f"class{name}{file_name}")
    plot_logists(pos, lims=50, file_path=file_path)
    return round(100.0 * acc1, 3), round(100.0 * acc2, 3)











