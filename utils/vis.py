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
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image
sys_args = helper.get_args()
colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])

#["blue", "red", "black", "violet", "orange"]
markers = ["o", "v", "s", "^", "d", "*", "D", "+", "k", "X"]
fontdict = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 30,
}



def plot_embedding(xy, file_path, lims=100, fontsize=30):
    plt.figure(figsize=(12, 12), dpi=160)
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


def scatter_points(data_dict, file_path, lims=100, fontsize=30):
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    plt.grid()

    for idx, (key, value) in enumerate(data_dict.items()):
        plt.scatter(value[:, 0], value[:, 1], lw=4, s=60, label=key, marker=markers[idx])
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


def plot_accuracy_dist_curve(atk_data, neg_data, steps, legends, path, fontsize=25, linewidth=5, markersize=25):
    plt.figure(figsize=(12, 12), dpi=200)
    plt.cla()
    plt.grid()
    global markers
    size = len(legends)
    num_steps = len(steps)
    fontdict["size"] = fontsize

    markers_neg = "P"
    if "ZEST" in legends[0]:
        markers = markers[2:]
        markers_neg = "X"

    for idx, (metric, data) in enumerate(atk_data.items()):
        plt.plot(data[:, 0], data[:, 1], linewidth=linewidth, linestyle='-', markersize=markersize, marker=markers[idx], c=colors[idx])
        # plot step info
        #plt.text(data[0, 0] + 5, data[0, 1] + 0.05, f"t=0", fontdict=fontdict)
        #plt.text(data[4, 0] + 5, data[4, 1] + 0.05, f"t={steps[4]}", fontdict=fontdict)
        #plt.text(data[-1, 0] + 5, data[-1, 1] + 0.05, f"t={steps[-1]}", fontdict=fontdict, color="black")

    for idx, (metric, data) in enumerate(neg_data.items()):
        legends += [f"{legends[idx]}(negative)"]
        plt.scatter(data[:, 0], data[:, 1], s=markersize**2, marker=markers_neg, c=colors[idx])

    #plt.legend(labels=legends, loc='best', fontsize=fontsize)
    #plt.xlim((80, 101))
    plt.ylim((0, 1.01))
    plt.xticks(np.arange(80, 100, 5))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Accuracy (%)", fontsize=fontsize)
    plt.ylabel("Distance", fontsize=fontsize)
    plt.savefig(path)
    print(f"-> saving fig: {path}")


def view_learning_state(data, file_path, fontsize=30):
    for key in data["keys"]:
        plt.figure(figsize=(16, 12), dpi=100)
        plt.cla()
        plt.grid()
        x = np.array(data["t"], dtype=np.int32)
        y = np.array(data[key])

        plt.plot(x, y, label=f"Surrogate Model", linewidth=5, marker="*", markersize=10, linestyle="solid")
        plt.plot(x, np.repeat(np.min(y), len(x)), color="black", linewidth=3, linestyle="dashdot")
        plt.text(float(np.argmin(y) * 20.0), np.min(y) - 3, round(float(np.min(y)), 2), fontsize=fontsize-5)
        plt.plot(x, np.repeat(np.max(y), len(x)), color="black", linewidth=3, linestyle="dashdot")
        plt.text(float(np.argmax(y) * 20.0), np.max(y) + 3, round(float(np.max(y)), 2), fontsize=fontsize-5)
        last_y = float(y[-1])
        plt.text(float(len(y)-1) * 20.0, last_y + 3, round(last_y, 2), fontsize=fontsize - 5)


        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel(key.upper(), fontsize=fontsize)
        if key == "acc":
            plt.ylim(0, 100.0)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="best", numpoints=1, fontsize=fontsize)
        fpath = file_path + f"_{key.upper()}.pdf"
        plt.savefig(fpath)
        print(f"-> saving fig: {fpath}")


def view_layer_activation(model_T, model_t, x, y, ori_x, target_layer, fig_path, size=8, fontsize=30, device=torch.device("cuda:0")):
    x, y = x.to(device), y.to(device)
    class_idx = y.tolist()
    model_0 = model_T.to(device)
    model_t = model_t.to(device)

    # extract LayerCAM for model_T
    CAM0 = LayerCAM(model_0, target_layer)
    cams = CAM0(class_idx=class_idx, scores=model_0(x))
    fused_cam0 = CAM0.fuse_cams(cams).detach().cpu()
    CAM0.remove_hooks()
    # extract LayerCAM for model_t
    CAM1 = LayerCAM(model_t, target_layer)
    cams = CAM1(class_idx=class_idx, scores=model_t(x))
    fused_camt = CAM1.fuse_cams(cams).detach().cpu()
    CAM1.remove_hooks()

    fig = plt.figure(figsize=(10, 4 + size * 2), dpi=200)
    img_idx = 0
    for idx in range(size):
        # original image
        img_idx += 1
        fig.add_subplot(size, 3, img_idx)
        pil_image = to_pil_image(ori_x[idx])
        plt.imshow(pil_image, interpolation='nearest')

        # overlay of target model output
        img_idx += 1
        fig.add_subplot(size, 3, img_idx)
        pil_map_T = to_pil_image(fused_cam0[idx].squeeze(0), mode='F')
        pil_overlay_T = overlay_mask(pil_image, pil_map_T, alpha=0.5)
        plt.imshow(pil_overlay_T, interpolation='nearest')

        # overlay of surrogate model output
        img_idx += 1
        fig.add_subplot(size, 3, img_idx)
        pil_map_t = to_pil_image(fused_camt[idx].squeeze(0), mode='F')
        pil_overlay_t = overlay_mask(pil_image, pil_map_t, alpha=0.5)
        plt.imshow(pil_overlay_t)
    pth = f"{fig_path}_OverLay.pdf"
    plt.savefig(pth)
    print(f"-> Saving OverLay:{pth}")

    fig = plt.figure(constrained_layout=True, figsize=(8, 4 + size * 3), dpi=200)
    # Left collum
    subfigs = fig.subfigures(1, 2, wspace=0.1, hspace=0.1)
    axsLeft = subfigs[0].subplots(size, 1)
    subfigs[0].set_facecolor('#ffffff')
    for nn, ax in enumerate(axsLeft):
        fmap = fused_cam0[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[0].suptitle(f'Model_T {target_layer}', fontsize=fontsize-2)
    subfigs[0].colorbar(pc, shrink=0.8, ax=axsLeft, location='bottom')
    # Right collum
    axsRight = subfigs[1].subplots(size, 1)
    subfigs[1].set_facecolor('#ffffff')
    for nn, ax in enumerate(axsRight):
        fmap = fused_camt[nn].squeeze(0).numpy()
        pc = pixel_plot(fmap, ax, hide_labels=True)
    subfigs[1].suptitle(f'Model_S {target_layer}', fontsize=fontsize-2)
    subfigs[1].colorbar(pc, shrink=0.8, ax=axsRight, location='bottom')
    fig.suptitle(f'LayerCAM of {target_layer}', fontsize=fontsize)
    pth = f"{fig_path}_ActativeLay.pdf"
    plt.savefig(pth)
    print(f"-> Saving ActativeLay:{pth}")


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
    data = {
        "model_T": xy[:int(len(xy) / 2)],
        "model_S": xy[int(len(xy) / 2):],
    }
    scatter_points(data, file_path=f"{fig_path}_t{step}.pdf", lims=100)


    select_logist_array = torch.cat(select_logist1 + select_logist2).detach().cpu().numpy()
    xy = TSNE(n_components=2, n_iter=1000, random_state=12345).fit_transform(select_logist_array)
    name = ""
    for k in select_t:
        name += f"{k}-"
    file_path = osp.join(fig_path, f"{fig_path}_C{name[:-1]}_t{step}.pdf")
    plot_embedding(xy, lims=90, file_path=file_path)
    return acc1, acc2











