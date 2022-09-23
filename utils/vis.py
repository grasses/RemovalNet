#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/06/28, homeway'


import torch
import numpy as np
import os.path as osp
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import helper
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



def plot_logist_embedding(teacher, student, test_loader, out_root, file_name, device=sys_args.device):
    teacher = teacher.to(device)
    student = student.to(device)
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

    select_logist_array = torch.cat(select_logist1 + select_logist2).numpy()
    pos = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(select_logist_array)

    name = "_"
    for k in select_t:
        name += f"{k}_"
    file_path = osp.join(out_root, f"class{name}{file_name}")
    plot_logists(pos, lims=50, file_path=file_path)
    return round(100.0 * acc1, 3), round(100.0 * acc2, 3)











